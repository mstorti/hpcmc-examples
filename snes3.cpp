/*$Id curspar-1.0.0-15-gabee420 Thu Jun 14 00:46:44 2007 -0300$*/

static char help[] = 
  "Newton's method to solve a combustion-like 1D problem.\n";

#include "petscsnes.h"

/* 
   User-defined routines
*/
extern int resfun(SNES,Vec,Vec,void*);
extern int jacfun(SNES,Vec,Mat*,Mat*,MatStructure*,void*);

struct SnesCtx {
  int N;
  double k,c,h,f0;
};

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  SNES         snes;         /* nonlinear solver context */
  PC           pc;           /* preconditioner context */
  KSP          ksp;          /* Krylov subspace method context */
  Vec          x,r;         /* solution, residual vectors */
  Mat          J;            /* Jacobian matrix */
  int          ierr,its,size;
  PetscScalar  pfive = .5,*xx;
  PetscTruth   flg;
  SnesCtx ctx;
  int N = 10;
  ctx.N = N;
  ctx.k = 0.1;
  ctx.c = 1;
  ctx.f0 = 0.5;
  ctx.h = 1.0/N;

  PetscInitialize(&argc,&argv,(char *)0,help);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size != 1) SETERRQ(1,"This is a uniprocessor example only!");

  // Create nonlinear solver context
  ierr = SNESCreate(PETSC_COMM_WORLD,&snes); CHKERRQ(ierr);
  ierr = SNESSetType(snes,SNESLS); CHKERRQ(ierr);

  // Create matrix and vector data structures;
  // set corresponding routines

  // Create vectors for solution and nonlinear function
  ierr = VecCreateSeq(PETSC_COMM_SELF,N+1,&x);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&r); CHKERRQ(ierr);
  double scal = 1.0;
  ierr = VecSet(x,scal); CHKERRQ(ierr);
  
  ierr = MatCreateMPIAIJ(PETSC_COMM_SELF,PETSC_DECIDE,
                         PETSC_DECIDE,N+1,N+1,
                         1,NULL,0,NULL,&J);CHKERRQ(ierr);

  ierr = SNESSetFunction(snes,r,resfun,&ctx); CHKERRQ(ierr);
  ierr = SNESSetJacobian(snes,J,J,jacfun,&ctx); CHKERRQ(ierr);

  ierr = SNESSolve(snes,NULL,x);CHKERRQ(ierr);
  Vec f;
  ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = SNESGetFunction(snes,&f,0,0);CHKERRQ(ierr);
  double rnorm;
  ierr = VecNorm(r,NORM_2,&rnorm);

  SNESGetLinearSolveIterations(snes,&its);
  ierr = PetscPrintf(PETSC_COMM_SELF,
                     "number of Newton iterations = "
                     "%d, norm res %g\n",
                     its,rnorm);CHKERRQ(ierr);

  ierr = VecDestroy(x);CHKERRQ(ierr);
  ierr = VecDestroy(r);CHKERRQ(ierr);
  ierr = SNESDestroy(snes);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}

double xc = 10;

/* ------------------------------------------------------------------- */
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
    ff[j] = -ctx.f0 + ctx.c*tanh(xxx/xc);
    ff[j] += ctx.k*(-xx[j+1]+2.0*xx[j]-xx[j-1])/(h*h);
  }

  ierr = VecRestoreArray(x,&xx);CHKERRQ(ierr);
  ierr = VecRestoreArray(f,&ff);CHKERRQ(ierr);

#if 1
  ierr = PetscPrintf(PETSC_COMM_SELF,"x:\n");
  ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"f:\n");
  ierr = VecView(f,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"================\n");
#endif
  return 0;
}

/* ------------------------------------------------------------------- */
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
    double chx = cosh(xxx/xc);
    A = ctx.c/(chx*chx)/xc;
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
  ierr = MatView(*jac,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr); 
  ierr = VecRestoreArray(x,&xx);CHKERRQ(ierr);
  return 0;
}
