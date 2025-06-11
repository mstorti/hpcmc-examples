static char help[] = 
  "Newton's method to solve a combustion-like 1D problem.\n";

#include <vector>
#include <cstdlib>
#include "petscsnes.h"
#include "H5Cpp.h"

using namespace std;

extern int resfun(SNES,Vec,Vec,void*);
// extern int jacfun(SNES,Vec,Mat*,Mat*,MatStructure*,void*);
extern PetscErrorCode jacfun(SNES,Vec,Mat,Mat,void *);

struct SnesCtx {
  int N;
  double k,c,h;
};

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

//---:---<*>---:---<*>---:---<*>---:---<*>---:---<*>
int h5petsc_vec_save(Vec x,const char *filename,const char *varname) {
  vector<double> vx;
  vec_gather(PETSC_COMM_WORLD,x,vx);
  H5::H5File file(filename,H5F_ACC_TRUNC);
  hsize_t n = vx.size();
  H5::DataSpace dataspace(1,&n);
  // Create the dataset.
  string svar(varname);
  H5::DataSet xdset =
    file.createDataSet(svar,H5::PredType::NATIVE_DOUBLE,dataspace);
  xdset.write(vx.data(),H5::PredType::NATIVE_DOUBLE);
  file.close();
  return 0;
}

PetscErrorCode snesmonitor(SNES snes, PetscInt its, PetscReal fnorm, void *ctx) {
  PetscPrintf(PETSC_COMM_WORLD, "SNES iter = %" PetscInt_FMT
              ",SNES Function norm %g\n",its,fnorm);
  return 0;
}

PetscErrorCode kspmonitor(KSP ksp, PetscInt n, PetscReal rnorm, void *dummy) {
  PetscPrintf(PETSC_COMM_WORLD, "KSP iter = %" PetscInt_FMT ",res %g\n",n,rnorm);
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  SNES         snes;         /* nonlinear solver context */
  PC           pc;           /* preconditioner context */
  KSP          ksp;          /* Krylov subspace method context */
  Vec          x,r,b0,b;         /* solution, residual vectors */
  Mat          J;            /* Jacobian matrix */
  int          ierr,its;
  PetscScalar  pfive = .5;
  PetscBool   flg;
  SnesCtx ctx;
  int N = 50;
  ctx.N = N;
  ctx.k = 1e-4;
  ctx.c = 1;
  ctx.h = 1.0/N;

  PetscInitialize(&argc,&argv,(char *)0,help);
  PetscLogDefaultBegin();
  int size, myrank;
  MPI_Comm_rank(PETSC_COMM_WORLD,&myrank);
  MPI_Comm_size(PETSC_COMM_WORLD,&size);
  if (size != 1) SETERRQ(PETSC_COMM_WORLD,1,"This is a uniprocessor example only!");

  // Create nonlinear solver context
  ierr = SNESCreate(PETSC_COMM_WORLD,&snes); CHKERRQ(ierr);
  ierr = SNESSetType(snes,SNESNEWTONLS); CHKERRQ(ierr);
  SNESGetKSP(snes, &ksp);
  KSPGetPC(ksp, &pc);
  PCSetType(pc,PCLU);
  //  KSPMonitorSet(ksp,kspmonitor,NULL,NULL);

#if 0
  SNESSetFromOptions(snes);
  double  abstol, rtol, stol;
  int maxit, maxf;
  SNESGetTolerances(snes,&abstol,&rtol,&stol,&maxit,&maxf);
  PetscPrintf(PETSC_COMM_WORLD,"atol=%g, rtol=%g, stol=%g, maxit=%D, maxf=%D\n",
              (double)abstol,(double)rtol,(double)stol,maxit,maxf);
#endif
  
  // Create matrix and vector data structures;
  // set corresponding routines

  int nnod = (N+1)*(N+1);
  // Create vectors for solution and nonlinear function
  ierr = VecCreateSeq(PETSC_COMM_SELF,nnod,&x);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&r); CHKERRQ(ierr);
  ierr = VecDuplicate(x,&b0); CHKERRQ(ierr);
  ierr = VecDuplicate(x,&b); CHKERRQ(ierr);

  VecSet(x,1.0);
#if 0
  double *xx;
  ierr = VecGetArray(x,&xx); CHKERRQ(ierr);
  for (int j=0; j<=N; j++) {
    xx[j]=((j%50)>25);
  }
  ierr = VecRestoreArray(x,&xx); CHKERRQ(ierr);
#endif
  
  ierr = MatCreateAIJ(PETSC_COMM_SELF,PETSC_DECIDE,
                      PETSC_DECIDE,nnod,nnod,
                      1,NULL,0,NULL,&J);CHKERRQ(ierr);
  ierr = MatSetOption(J, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE); CHKERRQ(ierr);

  ierr = SNESSetFunction(snes,r,resfun,&ctx); CHKERRQ(ierr);
  ierr = SNESSetJacobian(snes,J,J,jacfun,&ctx); CHKERRQ(ierr);
  Vec f;
  ierr = VecDuplicate(x,&f); CHKERRQ(ierr);
  
  SNESMonitorSet(snes,snesmonitor,NULL,NULL);

#if 0  
  SNESLineSearch linesearch;
  SNESGetLineSearch(snes,&linesearch);
  // SNESLineSearchSetType(linesearch, SNESLINESEARCHCP);
  SNESLineSearchSetType(linesearch, SNESLINESEARCHL2);
#endif
  
  // SNESLineSearchSetDefaultMonitor(linesearch,NULL);
  // PetscErrorCode SNESLineSearchMonitorSet(SNESLineSearch ls, PetscErrorCode (*f)(SNESLineSearch, void *), void *mctx, PetscErrorCode (*monitordestroy)(void **))
  // SNESLineSearchMonitorSet(linesearch,SNESLineSearchMonitor(linesearch,NULL);
  // SNESMonitorSet(snes,snesmonitor,NULL,NULL);

#if 1 // Use FD Jacobian
  ISColoring    iscoloring;
  MatFDColoring fdcoloring;
  MatColoring   coloring;

  /*
    This initializes the nonzero structure of the Jacobian. This is artificial
    because clearly if we had a routine to compute the Jacobian we wouldn't
    need to use finite differences.
  */
  jacfun(snes,x,J,J, &ctx);

  /*
    Color the matrix, i.e. determine groups of columns that share no common
    rows. These columns in the Jacobian can all be computed simultaneously.
  */
  MatColoringCreate(J, &coloring);
  MatColoringSetType(coloring,MATCOLORINGSL);
  MatColoringSetFromOptions(coloring);
  MatColoringApply(coloring, &iscoloring);
  MatColoringDestroy(&coloring);
  /*
    Create the data structure that SNESComputeJacobianDefaultColor() uses
    to compute the actual Jacobians via finite differences.
  */
  MatFDColoringCreate(J,iscoloring, &fdcoloring);
  MatFDColoringSetFunction(fdcoloring,(PetscErrorCode (*)(void))resfun,&ctx);
  MatFDColoringSetFromOptions(fdcoloring);
  MatFDColoringSetUp(J,iscoloring,fdcoloring);
  ISColoringDestroy(&iscoloring);

  /*
    Tell SNES to use the routine SNESComputeJacobianDefaultColor()
    to compute Jacobians.
  */
  SNESSetJacobian(snes,J,J,SNESComputeJacobianDefaultColor,fdcoloring);  
#endif
  
#if 1
  int nk=10000;
#else
  int nk=1;
  // double k1=1,k2=k1;
  double k1=0.01,k2=k1;
#endif
  // Compute the residual at position x
  resfun(snes,x,b0,&ctx);
  int krec = 0;
  for (int k=0; k<nk; k++) {
#if 0
    // Continuation in k
    double xi=(nk>1? double(k)/(nk-1) : 0);
    double logk =log(k1)+xi*log(k2/k1);
    ctx.k = exp(logk);
    PetscPrintf(PETSC_COMM_WORLD,"k %d, xi %f, k %f\n",k,xi,ctx.k);
#else
    // Continuation in the RHS
    VecCopy(b0,b); CHKERRQ(ierr);
    double alpha=1.0 - (nk>1? double(k)/(nk-1) : 0);
    alpha *= alpha;
    VecScale(b,alpha); CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD,"k %d, alpha %f\n",k,alpha);
#endif
    ierr = SNESSolve(snes,b,x); CHKERRQ(ierr);
#if 0  
    ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
#else
    if (k%10==0) {
      char fname[200];
      sprintf(fname,"./STEPS/tempo%d.h5",krec);
      h5petsc_vec_save(x,fname,"u");
      krec++;
    }
#endif

    resfun(snes,x,f,&ctx);
    double rnorm,xnorm,fnorm;
    ierr = VecNorm(f,NORM_2,&fnorm);
    ierr = VecNorm(r,NORM_2,&rnorm);
    ierr = VecNorm(x,NORM_2,&xnorm);
    SNESGetLinearSolveIterations(snes,&its);
    ierr = PetscPrintf(PETSC_COMM_SELF,"number of Newton iterations = "
                       "%d, norm res %g, norm x %g,norm f %g\n",
                       its,rnorm,xnorm,fnorm); CHKERRQ(ierr);
    if (rnorm>1e-3) break;
  }

  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&r);CHKERRQ(ierr);
  ierr = SNESDestroy(&snes);CHKERRQ(ierr);

  // PetscLogView(PETSC_VIEWER_STDOUT_WORLD);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}

//---:---<*>---:---<*>---:---<*>---:---<*>---:---<*>
#undef __FUNCT__
#define __FUNCT__ "resfun"
int resfun(SNES snes,Vec x,Vec f,void *data) {
  double *xx,*ff;
  SnesCtx &ctx = *(SnesCtx *)data;
  int ierr;
  double h = ctx.h;
  int N = ctx.N, N1=N+1, nnod = N1*N1;

  ierr = VecGetArray(x,&xx);CHKERRQ(ierr);
  ierr = VecGetArray(f,&ff);CHKERRQ(ierr);

  double X,Y;
  for (int j=0; j<=N; j++) {
    X = double(j)/N;
    for (int k=0; k<=N; k++) {
      Y = double(k)/N;
      int jk = k*(N+1)+j;
      if (j==0 || j==N || k==0 || k==N) {
        ff[jk] = xx[jk];
      } else {
        double xxx = xx[jk];
        ff[jk] = ctx.c*xxx*(0.5-xxx)*(1.0-xxx);
        ff[jk] += ctx.k*(-xx[jk-N1]-xx[jk+1]
                        +4.0*xx[jk]-xx[jk-1]-xx[jk+N1])/(h*h);
      }
    }
  }

  ierr = VecRestoreArray(x,&xx);CHKERRQ(ierr);
  ierr = VecRestoreArray(f,&ff);CHKERRQ(ierr);

#if 0
  ierr = PetscPrintf(PETSC_COMM_SELF,"x:\n");
  ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"f:\n");
  ierr = VecView(f,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"================\n");
#endif
  return 0;
}

//---:---<*>---:---<*>---:---<*>---:---<*>---:---<*>
#undef __FUNCT__
#define __FUNCT__ "jacfun"
int jacfun(SNES snes,Vec x,Mat jac,Mat jac1,void *data) {
  double *xx, A;
  SnesCtx &ctx = *(SnesCtx *)data;
  int ierr, j;
  int N = ctx.N, N1=N+1, nnod = N1*N1;
  double h=ctx.h, h2=h*h;

  ierr = VecGetArray(x,&xx);CHKERRQ(ierr);
  ierr = MatZeroEntries(jac);

  int cols[5];
  double coefs[5];
  coefs[0] = -ctx.k*1.0/h2;
  coefs[1] = -ctx.k*1.0/h2;
  coefs[3] = -ctx.k*1.0/h2;
  coefs[4] = -ctx.k*1.0/h2;
  for (int k=0; k<=N; k++) {
    for (int j=0; j<=N; j++) {
      int jk = k*(N+1)+j;
      if (j==0 || j==N || k==0 || k==N) {
        double A = 1.0;
        ierr = MatSetValues(jac,1,&jk,1,&jk,&A,
                            INSERT_VALUES); CHKERRQ(ierr);
      } else {
        double xxx = xx[j];
        double phidot = ctx.c * ((0.5-xxx)*(1.0-xxx)
                                 - xxx*(1.0-xxx) - xxx*(0.5-xxx));
        coefs[2] = ctx.k*4.0/h2 + phidot;
        cols[0] = jk-N1;
        cols[1] = jk-1;
        cols[2] = jk;
        cols[3] = jk+1;
        cols[4] = jk+N1;
        ierr = MatSetValues(jac,1,&jk,5,cols,coefs,INSERT_VALUES); 
      }
    }
  }
  ierr = MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = VecRestoreArray(x,&xx);CHKERRQ(ierr);

#if 0
  ierr = PetscPrintf(PETSC_COMM_SELF,"x:\n");
  ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"A:\n");
  ierr = MatView(*jac,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr); 
#endif

  return 0;
}

