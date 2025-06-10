static char help[] = 
  "Newton's method to solve a 2D network of springs.\n";

#include <cstdlib>
#include <iostream>
#include <vector>
#include <map>
#include <Eigen/Dense>
#include "petscsnes.h"

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

/* 
   User-defined routines
*/
int resfun(SNES,Vec,Vec,void*);
int jacfun_fd(SNES,Vec,Mat*,Mat*,MatStructure*,void*);
int jacfun_anly(SNES,Vec,Mat*,Mat*,MatStructure*,void*);

// Regularized Heaviside function based on COS, in interval [0,b]
double regheavis(double x,double b) {
  double xi=x/b;
  return xi<0.0? 0.0 : xi>1.0? 1.0 : 0.5*(1.0-cos(M_PI*xi));
}

// Regularized Heaviside function based on COS, in interval [a,b]
double regheavis(double x,double a,double b) {
  return regheavis(x-a,b-a);
}

//---:---<*>---:---<*>---:---<*>---:---<*>---:---<*>
class spring_t {
public:
  virtual double force(double len)=0;
};

//---:---<*>---:---<*>---:---<*>---:---<*>---:---<*>
class lin_spring_t : public spring_t {
public:
  double force(double len) {
    // return sqrt(len);
    return len;
  }
} lin_spring;

//---:---<*>---:---<*>---:---<*>---:---<*>---:---<*>
class force_t {
public:
  virtual void force(VectorXd &x,VectorXd &force)=0;
};

//---:---<*>---:---<*>---:---<*>---:---<*>---:---<*>
class grav_t : public force_t {
public:
  double w;
  void force(VectorXd &x,VectorXd &force) {
    force(2) = -w;
  }
  grav_t() : w(0.0) {}
} grav;

//---:---<*>---:---<*>---:---<*>---:---<*>---:---<*>
class sphere_force_t : public force_t {
public:
  double R,dr,k;
  VectorXd xc;
  void force(VectorXd &x,VectorXd &force) {
    VectorXd dx = x-xc;
    // Distance to center
    double r = dx.norm();
    force = -(1.0-regheavis(r,0,R))*k*dx/r;
  }
} sphere_force;

double sqr(double x) { return x*x; }

//---:---<*>---:---<*>---:---<*>---:---<*>---:---<*>
class ellipsoid_t : public force_t {
public:
  double dr,k;
  VectorXd axis,xc;
  void force(VectorXd &x,VectorXd &force) {
    VectorXd dx = x-xc;
    // Distance to center
    const int ndim=3;
    VectorXd nor(ndim);
    double rho=0.0;
    for (int j=0; j<ndim; j++) {
      nor(j) = dx(j)/sqr(axis(j));
      rho += sqr(dx(j)/axis(j));
    }
    // force = -(1.0-regheavis(rho,1-rho,1+rho))*k*nor/dx.norm();
    force = -(1.0-regheavis(rho,1-rho,1+rho))*k*nor/dx.norm();
  }
} ellipsoid;

//---:---<*>---:---<*>---:---<*>---:---<*>---:---<*>
class field_t : public force_t {
public:
  void force(VectorXd &x,VectorXd &force) {
    // Attraction to the z=0.1 plane
    force.fill(0.0);
    force(2) = 1*(x(2)-0.1);
  }
} field;

//---:---<*>---:---<*>---:---<*>---:---<*>---:---<*>
struct snes_ctx_t {
  // N := number of segments per side
  // N1 := N+1, number of nodes per side
  // ndim := number of dimensions
  // nnod := number of nodes
  // nelem := number of elements (quads)
  // neq := number of equations
  // nstep := number of steps to solve
  int N,N1,ndim,nnod,nelem,neq,nstep,nsteprlx,nstepz;
  // h := Step size, length of the segments in the ref mesh
  // L0 := length of side in the ref mesh
  // DL := elongation of right side
  // p := power in the stiffness law
  double h,L0,DL,p;
  // Call back function that defines relative stifness coeff
  double stiff(double x) { return 1.0; }
  // Initial position of the nodes
  vector<double> xref;
  // (node,dof) -> fixed displs
  map< pair<int,int>,double> bcfix;
  // Vector and scatter to gather all values in the global vector
  Vec uloc;
  VecScatter scat;
  // Utility function to set the value of a node and dof
  // pair to a value
  void set_bc(int node,int dof,double val) {
    bcfix[pair<int,int>(node,dof)] = val;
 }
  // Spring
  spring_t *springp;
  // Force field
  force_t *forcep;
  // Initialize the problem
  void init();
  // Set the initial position vector
  double xinit(int node,int dof);
  // Set the boundary conditions 
  void setup_step(int step);
  // Compute the resfun for the SNES
  int resfun(Vec x,Vec f);
  // Compute the jacfun for the SNES
  int jacfun(Vec x0,Mat* jac);
  // Ctor
  snes_ctx_t(); 
  // Dtor
  ~snes_ctx_t(); 
};

//---:---<*>---:---<*>---:---<*>---:---<*>---:---<*>
snes_ctx_t::snes_ctx_t() : uloc(NULL), scat(NULL),
                           springp(NULL), forcep(NULL) {}

//---:---<*>---:---<*>---:---<*>---:---<*>---:---<*>
snes_ctx_t::~snes_ctx_t() {
  if (uloc) {
    VecDestroy(&uloc);
    VecScatterDestroy(&scat);
  }
}

//---:---<*>---:---<*>---:---<*>---:---<*>---:---<*>
void snes_ctx_t::init() {

  ndim = 3;
  N = 120;
  L0 = 1;
  p = 1;
  nsteprlx = 1;
  nstepz = 400;
  nstep = nsteprlx*nstepz;
  grav_t &g = grav;
  g.w = -0.01;
  // forcep = &g;
  sphere_force_t &f = sphere_force;
  f.R = 0.25;
  f.xc.resize(ndim);
  cout << "xc " << f.xc.size() << endl;
  f.xc << 0.5*L0,0.5*L0,-1.1*f.R;
  f.dr= 0.5*f.R;
  f.k = 0.01;
  forcep = &f;
  // Ellipsoid
  ellipsoid_t &e = ellipsoid;
  e.axis.resize(ndim);
  e.xc.resize(ndim);
  double R = 0.25;
  e.axis << R,R,R;
  e.xc << 0.5*L0,0.5*L0,0.5*R;
  e.dr = 0.05;
  e.k = 10;
  
  // Special force field for debugging
  // forcep = &field;
#if 0
  // Explore the shape of the force field
  VectorXd x(ndim),ff(ndim);
  x << 0.5,0.5,0;
  int N = 300;
  for (int k=0; k<N; k++) {
    x(2) = -0.5+1*double(k)/N;
    forcep->force(x,ff);
    printf("z %f f %f %f %f\n",x(2),ff(0),ff(1),ff(2));
  }
  exit(0);
#endif
  N1 = N+1;
  h = 1.0/N;
  DL = 0.5;
  springp = &lin_spring;

  nnod = N1*N1;
  nelem = N*N;
  neq = nnod*ndim;

  xref.resize(neq);
  for (int j=0; j<N1; j++) {
    double x=j*h;
    for (int i=0; i<N1; i++) {
      double y=i*h;
      int node = i*N1+j;
      xref[node*ndim+0] = x;
      xref[node*ndim+1] = y;
      xref[node*ndim+2] = 0.0;
    }
  }
}

//---:---<*>---:---<*>---:---<*>---:---<*>---:---<*>
// Set the boundary conditions 
void snes_ctx_t::setup_step(int step) {
  // Boundaries BCs
  // x=0 side to DX=0, x=1 side to DX=0.1, (y=0,y=1 -> DY=0)
  // Nodes are numbered first by X and then by Y, so node
  // (i*h,j*h) -> is node=i*(N+1)+j
#if 1
  for (int j=0; j<=N; j++) {
    // Right side set to DX=0,DY=0
    int node = j*N1;
    set_bc(node,0,0.0);
    set_bc(node,2,0.0);
    // Left side set to DX=DL,DY=0
    node = j*N1+N;
    set_bc(node,0,L0);
    set_bc(node,2,0.0);
    // Bottom side
    node = j;
    set_bc(node,1,0.0);
    set_bc(node,2,0.0);
    // Bottom side
    node = N*N1+j;
    set_bc(node,1,L0);
    set_bc(node,2,0.0);
  }
  sphere_force_t &s = sphere_force;
  if (step%nsteprlx==0) {
    // s.dr= 0.5*s.R;
    s.dr= 0.1*s.R;
    s.k = 10;
    // double zc = s.xc(2)+0.005*s.R; 
    double zc = s.xc(2) + (4*s.R)/nstepz;
    if (1 || zc<0.145) s.xc(2) = zc;

    double 
      Rorbit=0.0,
      Dt=0.01,
      T=1,
      omega=2*M_PI/T,
      time = (step/nsteprlx)*Dt;
    s.xc(0) = 0.5*L0+Rorbit*cos(omega*time);
    s.xc(1) = 0.5*L0+Rorbit*sin(omega*time);
  } else { 
    s.dr *= 0.9;
    s.k *= 2;
  }
  printf("setup step %d, dr %g, k %g, zc %f\n",
         step,s.dr,s.k,s.xc(2));
#elif 0
  double xi = -0.5+double(step)/nstep;
  for (int j=0; j<=N; j++) {
    // Right side set to DX=0,DY=0
    int node = j*N1;
    set_bc(node,0,0.0);
    set_bc(node,1,xinit(node,1));
    set_bc(node,2,0.0);
    // Left side set to DX=DL,DY=0
    node = j*N1+N;
    set_bc(node,0,L0+xi*DL);
    set_bc(node,1,xinit(node,1));
    set_bc(node,2,0.0);
  }
#else
  VectorXd X(2),XC(2);
  XC.fill(0.5*L0);
  for (int node=0; node<nnod; node++) {
    double
      x = xinit(node,0),
      y = xinit(node,1);
    double tol=1e-6;
    if (x<tol || x>L0-tol || y<tol || y>L0-tol) {
      X << x,y;
      X -= XC;
      X /= X.norm();
      X += XC;
      set_bc(node,0,X(0));
      set_bc(node,1,X(1));
      set_bc(node,2,0.0);
    }
  }
#endif
}

//---:---<*>---:---<*>---:---<*>---:---<*>---:---<*>
double snes_ctx_t::xinit(int node,int dof) {
  return xref[node*ndim+dof];
}

// Given a distributed PETSc vector `vec' gather all the
// ranges in all processor in a full vector of doubles
// `values'.  This full vector is available in all the
// processors.
// WARNING: this may be inefficient and non
// scalable, it is just a dirty trick to have access to all
// the values of the vectors in all the processor.
// Usage:
// Vec v;
// // ... create and fill v eith values at each processor
// // ... do the Assembly
// vector<double> values;
// vec_gather(MPI_COMM_WORLD,v,values);
// //... now you have all the elements of `v' in `values'
void vec_gather(MPI_Comm comm,Vec v,vector<double> &values) {
  // n: global size of vector
  // nlocal: local (PETSc) size
  int n,nlocal;
  // Get the global size
  VecGetSize(v,&n);
  // Resize the local buffer
  values.clear();
  values.resize(n,0.0);
  // Get the local size
  VecGetLocalSize(v,&nlocal);

  // Gather all the local sizes in order to compute the
  // counts and displs for the Allgatherv
  int size, myrank;
  MPI_Comm_rank(comm,&myrank);
  MPI_Comm_size(comm,&size);
  vector<int> counts(size),displs(size);
  MPI_Allgather(&nlocal,1,MPI_INT,
                &counts[0],1,MPI_INT,comm);
  displs[0]=0;
  for (int j=1; j<size; j++)
    displs[j] = displs[j-1] + counts[j-1];

  // Get the internal values of the PETSc vector
  double *vp;
  VecGetArray(v,&vp);
  // Do the Allgatherv to the local vector
  MPI_Allgatherv(vp,nlocal,MPI_DOUBLE,
                 &values[0],&counts[0],&displs[0],MPI_DOUBLE,comm);
  // Restore the array
  VecRestoreArray(v,&vp);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv) {
  SNES         snes;         /* nonlinear solver context */
  Vec          x,r;         /* solution, residual vectors */
  Mat          J;            /* Jacobian matrix */
  int          ierr,its,size;
  snes_ctx_t ctx;

  PetscInitialize(&argc,&argv,(char *)0,help);

  ctx.init();

  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size != 1) SETERRQ(PETSC_COMM_WORLD,1,
                         "This is a uniprocessor example only!");

  // Create nonlinear solver context
  ierr = SNESCreate(PETSC_COMM_WORLD,&snes); CHKERRQ(ierr);
  ierr = SNESSetType(snes,SNESLS); CHKERRQ(ierr);
  SNESSetFromOptions(snes);
  double  abstol, rtol, stol;
  int maxit, maxf;
  SNESGetTolerances(snes,&abstol,&rtol,&stol,&maxit,&maxf);
  SNESMonitorSet(snes,SNESMonitorDefault,NULL, NULL);

  PetscPrintf(PETSC_COMM_WORLD,
              "atol=%g, rtol=%g, stol=%g, maxit=%D, maxf=%D\n",
              (double)abstol,(double)rtol,(double)stol,maxit,maxf);
 
  // Create vectors for solution and nonlinear function
  ierr = VecCreateSeq(PETSC_COMM_SELF,ctx.neq,&x);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&r); CHKERRQ(ierr);

#if 0
  // Create the scatter and the target vector
  VecScatterCreateToZero(u,ctx.scat,ctx.uloc);
  // Do the scatter
  VecScatterBegin(scat,u,uloc,INSERT_VALUES,SCATTER_FORWARD);
  VecScatterEnd(scat,u,uloc,INSERT_VALUES,SCATTER_FORWARD);
#endif
  
  double *xx;
  int ndim = ctx.ndim;
  ierr = VecGetArray(x,&xx); CHKERRQ(ierr);
  for (int j=0; j<ctx.nnod; j++) 
    for (int k=0; k<ndim; k++) 
      xx[j*ndim+k] = ctx.xinit(j,k);
  ierr = VecRestoreArray(x,&xx); CHKERRQ(ierr);

  ierr = MatCreateMPIAIJ(PETSC_COMM_SELF,PETSC_DECIDE,
                         PETSC_DECIDE,ctx.neq,ctx.neq,
                         27,NULL,0,NULL,&J);CHKERRQ(ierr);

  ierr = SNESSetFunction(snes,r,resfun,&ctx); CHKERRQ(ierr);
  // ierr = SNESSetJacobian(snes,J,J,jacfun_fd,&ctx); CHKERRQ(ierr);
  ierr = SNESSetJacobian(snes,J,J,jacfun_anly,&ctx); CHKERRQ(ierr);

#if 0
  // resfun(snes,x,r,&ctx);
  jacfun_fd(snes,x,&J,&J,NULL,&ctx);
  // MatView(J,PETSC_VIEWER_STDOUT_SELF); CHKERRQ(ierr);

  const int *colsp;
  const double *valsp;
  int ncols;
  for (int j=0; j<ctx.neq; j++) {
    MatGetRow(J,j,&ncols,&colsp,&valsp);
    for (int l=0; l<ncols; l++) 
      printf("%d %d %f\n",j,colsp[l],valsp[l]);
  }
  
  exit(0);
#endif

  for (int j=0; j<ctx.nstep; j++) {
    ctx.setup_step(j);
    ierr = SNESSolve(snes,NULL,x);CHKERRQ(ierr);
  
#if 1
    // Use HDF5
    PetscObjectSetName((PetscObject)x,"u");
    PetscViewer viewer;
    char filename[1000];
    sprintf(filename,"./states/u%d.h5",j);
    ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD,filename,
                               FILE_MODE_WRITE,&viewer); CHKERRQ(ierr);
    ierr = VecView(x,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
#else
    // Write to stdout
    ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
#endif
  
    Vec f;
    ierr = SNESGetFunction(snes,&f,0,0);CHKERRQ(ierr);
    double rnorm;
    ierr = VecNorm(r,NORM_2,&rnorm);

    SNESGetLinearSolveIterations(snes,&its);
    ierr = PetscPrintf(PETSC_COMM_SELF,
                       "number of Newton iterations = "
                       "%d, norm res %g\n",
                       its,rnorm);CHKERRQ(ierr);
    if (rnorm>1e-7) {
      printf("aborting due to large error rnorm %g\n",rnorm);
      exit(0);
    }
  }

  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&r);CHKERRQ(ierr);
  ierr = SNESDestroy(&snes);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}

//---:---<*>---:---<*>---:---<*>---:---<*>---:---<*>
#undef __FUNCT__
#define __FUNCT__ "resfun"
int resfun(SNES snes,Vec x,Vec f,void *data) {
  snes_ctx_t &ctx = *(snes_ctx_t *)data;
  return ctx.resfun(x,f);
}

//---:---<*>---:---<*>---:---<*>---:---<*>---:---<*>
int snes_ctx_t::resfun(Vec x,Vec f) {
  int ierr;
  VecSet(f,0.0);
  double *xx,*ff;
  ierr = VecGetArray(x,&xx);CHKERRQ(ierr);
  ierr = VecGetArray(f,&ff);CHKERRQ(ierr);

  // node0,node1 are the nodes at the extreme of the spring
  // xref{0,1} are the ref positions of the nodes
  // x{0,1} are the current positions of the nodes
  VectorXd xref0(ndim),xref1(ndim),force(ndim);
  VectorXd x0(ndim),x1(ndim),dx;
  double *xp;
  for (int i0=0; i0<N1; i0++) {
    for (int j0=0; j0<N1; j0++) {
      // Node at one extreme of the spring
      int node0=i0*N1+j0;
      // Reference position of node0
      xp = &xref[node0*ndim];
      xref0 << xp[0],xp[1],xp[2];
      // Current position of node0
      xp = &xx[node0*ndim];
      x0 << xp[0],xp[1],xp[2];
      // Force field at node 0
      forcep->force(x0,force);
      for (int l=0; l<ndim; l++) 
        ff[node0*ndim+l] += force(l);
      for (int l=0; l<4; l++) {
        int i1=i0,j1=j0;
        // node1 = EAST node
        if (l==0) i1++;
        // node1 = NORTH node
        else if (l==1) j1++;
        // node1 = NORTH-EAST node
        else if (l==2) { i1++; j1++; }
        // node1 = SOUTH-EAST node
        else if (l==3) { i1++; j1--; }

        if (i1<0 || i1>=N1 || j1<0 || j1>=N1) continue;
        
        int node1=i1*N1+j1;
        // if (!(node0==6 && node1==7)) continue;
        // Reference position of node1
        xp = &xref[node1*ndim];
        xref1 << xp[0],xp[1],xp[2];
        // Current position of node1
        xp = &xx[node1*ndim];
        x1 << xp[0],xp[1],xp[2];

        dx = x1-x0;
        // Force of this element
        double len = dx.norm();
        double ffj = springp->force(len)/len;
        // printf("node %d-%d len %f len0 %f ffj %f\n",node0,node1,len,len0,ffj);
        // Unit vector in the direction of the spring
        for (int l=0; l<ndim; l++) {
          double w = ffj*dx(l);
          ff[node0*ndim+l] -= w;
          ff[node1*ndim+l] += w;
        }
      }
    }
  }

  for (auto &q : bcfix) {
    int node = q.first.first;
    int dof = q.first.second;
    double val = q.second;
    int jeq = node*ndim+dof;
    ff[jeq] = xx[jeq]-val;
  }
  
  ierr = VecRestoreArray(x,&xx);CHKERRQ(ierr);
  ierr = VecRestoreArray(f,&ff);CHKERRQ(ierr);
  double fnorm;
  ierr = VecNorm(f,NORM_2,&fnorm);
  
  return 0;
}

//---:---<*>---:---<*>---:---<*>---:---<*>---:---<*>
// This is a wrapper, just calls the snes_ctx_t method
#undef __FUNCT__
#define __FUNCT__ "jacfun"
int jacfun_fd(SNES snes,Vec x0,Mat *jac,Mat*,
              MatStructure *,void *data) {
  int ierr;
  // x0 is the reference x, x1 is perturbed
  // f0,f1 are the residuals at those states
  Vec x1,f0,f1;
  ierr = VecDuplicate(x0,&x1); CHKERRQ(ierr);
  ierr = VecDuplicate(x0,&f0); CHKERRQ(ierr);
  ierr = VecDuplicate(x0,&f1); CHKERRQ(ierr);

  ierr = MatZeroEntries(*jac); CHKERRQ(ierr);
  double *x1p,*f0p,*f1p;
  double epsil=1e-7;
  int neq;
  ierr = VecGetSize(x0,&neq); CHKERRQ(ierr);

  resfun(snes,x0,f0,data);
      
  for (int k=0; k<neq; k++) {
    ierr = VecCopy(x0,x1); CHKERRQ(ierr);
    ierr = VecGetArray(x1,&x1p); CHKERRQ(ierr);
    x1p[k] += epsil;
    ierr = VecRestoreArray(x1,&x1p); CHKERRQ(ierr);
    resfun(snes,x1,f1,data);

    ierr = VecGetArray(f0,&f0p); CHKERRQ(ierr);
    ierr = VecGetArray(f1,&f1p); CHKERRQ(ierr);
    double tol=1e-10;
    vector<int> indx;
    vector<double> coef;
    // printf("J(%d,:) ",k);
    for (int j=0; j<neq; j++) {
      if (fabs(f1p[j]-f0p[j])>tol) {
        indx.push_back(j);
        double c = (f1p[j]-f0p[j])/epsil;
        coef.push_back(c);
        // printf("(%d,%f) ",j,c);
      }
    }
    // printf("\n");
    ierr = VecRestoreArray(f0,&f0p); CHKERRQ(ierr);
    ierr = VecRestoreArray(f1,&f1p); CHKERRQ(ierr);
    MatSetValues(*jac,indx.size(),indx.data(),1,&k,
                 coef.data(),INSERT_VALUES); CHKERRQ(ierr);
  }
  // Assembly the matrix
  ierr = MatAssemblyBegin(*jac,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*jac,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  // Destroy auxiliary vectors
  ierr = VecDestroy(&x1); CHKERRQ(ierr);
  ierr = VecDestroy(&f0); CHKERRQ(ierr);
  ierr = VecDestroy(&f1); CHKERRQ(ierr);
  
  return 0;
}

//---:---<*>---:---<*>---:---<*>---:---<*>---:---<*>
#undef __FUNCT__
#define __FUNCT__ "jacfun_anly"
int jacfun_anly(SNES snes,Vec x0,Mat* jac,Mat*,
              MatStructure *,void *data) {
  snes_ctx_t &ctx = *(snes_ctx_t *)data;
  return ctx.jacfun(x0,jac);
}

//---:---<*>---:---<*>---:---<*>---:---<*>---:---<*>
// Computes the Jacobian of the residual vector
// with respect to the state vector.
int snes_ctx_t::jacfun(Vec x,Mat* jac) {

  int ierr;
  // ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  double *xx;
  ierr = VecGetArray(x,&xx);CHKERRQ(ierr);

  // node0,node1 are the nodes at the extreme of the spring
  // xref{0,1} are the ref positions of the nodes
  // x{0,1} are the current positions of the nodes
  VectorXd xref0(ndim),xref1(ndim);
  VectorXd x0(ndim),x1(ndim),dx,tan;
  MatrixXd jac1(2*ndim,2*ndim),F11(ndim,ndim);
  MatrixXd Id = MatrixXd::Identity(ndim,ndim);
  double *xp;
  vector<int> indx(6);
  VectorXd f0(ndim),fp(ndim),fm(ndim);
  MatrixXd fjac(ndim,ndim);

  ierr = MatZeroEntries(*jac); CHKERRQ(ierr);
  for (int i0=0; i0<N1; i0++) {
    for (int j0=0; j0<N1; j0++) {
      // Node at one extreme of the spring
      int node0=i0*N1+j0;
      // Reference position of node0
      xp = &xref[node0*ndim];
      xref0 << xp[0],xp[1],xp[2];
      // Current position of node0
      xp = &xx[node0*ndim];
      x0 << xp[0],xp[1],xp[2];

      fjac.fill(0.0);
      // Jacobian of the force field at node 0
      double epsln=1e-5;
      for (int l=0; l<ndim; l++) {
        indx[l] = node0*ndim+l;
        x1 = x0;
        x1(l) += epsln;
        forcep->force(x1,fp);
        x1(l) -= 2*epsln;
        forcep->force(x1,fm);
        fjac.col(l) = (fp-fm)/(2*epsln);
      }
      fjac.transposeInPlace();
      MatSetValues(*jac,3,indx.data(),3,indx.data(),
                   fjac.data(),ADD_VALUES); CHKERRQ(ierr);
      
      for (int l=0; l<4; l++) {
        int i1=i0,j1=j0;
        // node1 = EAST node
        if (l==0) i1++;
        // node1 = NORTH node
        else if (l==1) j1++;
        // node1 = NORTH-EAST node
        else if (l==2) { i1++; j1++; }
        // node1 = SOUTH-EAST node
        else if (l==3) { i1++; j1--; }

        if (i1<0 || i1>=N1 || j1<0 || j1>=N1) continue;
        
        int node1=i1*N1+j1;
        // Reference position of node1
        xp = &xref[node1*ndim];
        xref1 << xp[0],xp[1],xp[2];
        // Current position of node1
        xp = &xx[node1*ndim];
        // printf("xx[node1,:] %g %g %g\n",
        //        xx[node1*ndim],xx[node1*ndim+1],xx[node1*ndim+2]);
        x1 << xp[0],xp[1],xp[2];
        // cout << "xref0 " << endl << xref0 << endl;
        // cout << "xref1 " << endl << xref1 << endl;
        // cout << "x0 " << endl << x0 << endl;
        // cout << "x1 " << endl << x1 << endl;

        dx = x1-x0;
        double len = dx.norm();
        // printf("node %d-%d len %f len0 %f\n",
        //        node0,node1,len,len0);
        // Unit vector in the direction of the spring
        tan = dx/len;
        double Force = springp->force(len);
        double epsln = 1e-5;
        double Fp,Fm;
        Fp = springp->force(len+epsln);
        Fm = springp->force(len-epsln);
        double Fdot = (Fp-Fm)/(2*epsln);
        double fdot = Fdot/len-Force/(len*len);
        F11 = fdot*len*tan*tan.transpose() + Force/len*Id;
        // cout << "F11: " << endl << F11 << endl;
        jac1.block(0,0,ndim,ndim) = F11;
        jac1.block(0,ndim,ndim,ndim) = -F11;
        jac1.block(ndim,0,ndim,ndim) = -F11;
        jac1.block(ndim,ndim,ndim,ndim) = F11;
        for (int j=0; j<ndim; j++) {
          indx[j] = node0*ndim+j;
          indx[ndim+j] = node1*ndim+j;
        }
        MatSetValues(*jac,6,indx.data(),6,indx.data(),
                     jac1.data(),ADD_VALUES); CHKERRQ(ierr);
      }
    }
  }
  // Assembly the matrix
  ierr = MatAssemblyBegin(*jac,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*jac,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  vector<int> bcdofs;
  for (auto &q : bcfix) {
    int node = q.first.first;
    int dof = q.first.second;
    bcdofs.push_back(node*ndim+dof);
  }
  MatZeroRows(*jac,bcdofs.size(),bcdofs.data(),1.0,NULL,NULL);

  // MatView(*jac,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  
  ierr = VecRestoreArray(x,&xx);CHKERRQ(ierr);
  return 0;
}
