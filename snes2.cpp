static char help[] = 
  "Newton's method to solve a combustion-like 1D problem.\n";

#include <vector>
#include <cstdlib>
#include "petscsnes.h"
#include "H5Cpp.h"

using namespace std;

/* 
   User-defined routines
*/
extern int resfun(SNES,Vec,Vec,void*);
extern int jacfun(SNES,Vec,Mat*,Mat*,MatStructure*,void*);

struct SnesCtx {
  int N;
  double k,c,h,f0;
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

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  SNES         snes;         /* nonlinear solver context */
  Vec          x,r;         /* solution, residual vectors */
  Mat          J;            /* Jacobian matrix */
  int          ierr,its,size;
  SnesCtx ctx;
  int N = 100;

  PetscInitialize(&argc,&argv,(char *)0,help);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-N",&N,PETSC_NULL);CHKERRQ(ierr);

  ctx.N = N;
  ctx.k = 0.001;
  ctx.c = 1;
  ctx.f0 = 0.0;
  ctx.h = 1.0/N;
  
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size != 1) SETERRQ(PETSC_COMM_WORLD,1,"This is a uniprocessor example only!");

  // Create nonlinear solver context
  ierr = SNESCreate(PETSC_COMM_WORLD,&snes); CHKERRQ(ierr);
  ierr = SNESSetType(snes,SNESLS); CHKERRQ(ierr);
  SNESSetFromOptions(snes);
  double  abstol, rtol, stol;
  int maxit, maxf;
  SNESGetTolerances(snes,&abstol,&rtol,&stol,&maxit,&maxf);
  PetscPrintf(PETSC_COMM_WORLD,"atol=%g, rtol=%g, stol=%g, maxit=%D, maxf=%D\n",
              (double)abstol,(double)rtol,(double)stol,maxit,maxf);

  // Create matrix and vector data structures;
  // set corresponding routines

  // Create vectors for solution and nonlinear function
  ierr = VecCreateSeq(PETSC_COMM_SELF,N+1,&x);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&r); CHKERRQ(ierr);

  double *xx;
  ierr = VecGetArray(x,&xx); CHKERRQ(ierr);
  for (int j=0; j<=N; j++) {
    xx[j]=((j%50)>25);
  }
  ierr = VecRestoreArray(x,&xx); CHKERRQ(ierr);
  
  ierr = MatCreateMPIAIJ(PETSC_COMM_SELF,PETSC_DECIDE,
                         PETSC_DECIDE,N+1,N+1,
                         1,NULL,0,NULL,&J);CHKERRQ(ierr);

  ierr = SNESSetFunction(snes,r,resfun,&ctx); CHKERRQ(ierr);
  ierr = SNESSetJacobian(snes,J,J,jacfun,&ctx); CHKERRQ(ierr);

  ierr = SNESSolve(snes,NULL,x);CHKERRQ(ierr);

#if 0
  // Doesn't work
  PetscObjectSetName((PetscObject)x,"temp");
  PetscViewer viewer;
  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD,"./temp.h5",
			     FILE_MODE_WRITE,&viewer); CHKERRQ(ierr);
  ierr = VecView(x,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
#elif 1
  // Use the function defined above
  h5petsc_vec_save(x,"temp.h5","u");
#else
  // Just plain output to stdout
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

  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&r);CHKERRQ(ierr);
  ierr = SNESDestroy(&snes);CHKERRQ(ierr);

  PetscLogView(PETSC_VIEWER_STDOUT_WORLD);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}

//---:---<*>---:---<*>---:---<*>---:---<*>---:---<*>
#undef __FUNCT__
#define __FUNCT__ "resfun"
int resfun(SNES snes,Vec x,Vec f,void *data)
{
  double *xx,*ff;
  SnesCtx &ctx = *(SnesCtx *)data;
  int ierr;
  double h = ctx.h;

  ierr = VecGetArray(x,&xx);CHKERRQ(ierr);
  ierr = VecGetArray(f,&ff);CHKERRQ(ierr);

  ff[0] = xx[0];
  ff[ctx.N] = xx[ctx.N];

  for (int j=1; j<ctx.N; j++) {
    double xxx = xx[j];
    ff[j] = ctx.f0 + ctx.c*xxx*(0.5-xxx)*(1.0-xxx);
    ff[j] += ctx.k*(-xx[j+1]+2.0*xx[j]-xx[j-1])/(h*h);
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
int jacfun(SNES snes,Vec x,Mat* jac,Mat* jac1,
           MatStructure *flag,void *data) {
  double *xx, A;
  SnesCtx &ctx = *(SnesCtx *)data;
  int ierr, j;

  ierr = VecGetArray(x,&xx);CHKERRQ(ierr);
  ierr = MatZeroEntries(*jac);

  j=0; A = 1;
  ierr = MatSetValues(*jac,1,&j,1,&j,&A,
                      INSERT_VALUES); CHKERRQ(ierr);

  j=ctx.N; A = 1;
  ierr = MatSetValues(*jac,1,&j,1,&j,&A,
                      INSERT_VALUES); CHKERRQ(ierr);

  int cols[3];
  double coefs[3];
  double h=ctx.h, h2=h*h;
  coefs[0] = -ctx.k*1.0/h2;
  coefs[1] = +ctx.k*2.0/h2;
  coefs[2] = -ctx.k*1.0/h2;
  for (j=1; j<ctx.N; j++) {
    double xxx = xx[j];
    A = ctx.c * ((0.5-xxx)*(1.0-xxx) - xxx*(1.0-xxx) - xxx*(0.5-xxx));
    ierr = MatSetValues(*jac,1,&j,1,&j,&A,INSERT_VALUES); 
    // ff[j] += ctx.k*(-xx[j+1]+2.0*xx[j+1]-xx[j-1])/(h*h);
    cols[0] = j-1;
    cols[1] = j;
    cols[2] = j+1;
    ierr = MatSetValues(*jac,1,&j,3,cols,coefs,ADD_VALUES); 
    
    CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(*jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  // ierr = MatView(*jac,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr); 
  // printf("en jacfun\n");
  ierr = VecRestoreArray(x,&xx);CHKERRQ(ierr);
  return 0;
}
