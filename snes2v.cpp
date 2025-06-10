static char help[] =
  "Newton's method to solve a combustion-like 2D problem.\n";

#include <vector>
#include <cstdlib>
#include "petscsnes.h"
#include "H5Cpp.h"

using namespace std;

/*
   User-defined routines
*/
extern int resfun (SNES, Vec, Vec, void *);
extern int jacfun (SNES, Vec, Mat *, Mat *, MatStructure *, void *);

struct SnesCtx
{
  int N;
  double k, c, h, f0;
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
void
vec_gather (MPI_Comm comm, Vec v, vector < double >&values)
{
  // n: global size of vector
  // nlocal: local (PETSc) size
  int n, nlocal;
  // Get the global size
  VecGetSize (v, &n);
  // Resize the local buffer
  values.clear ();
  values.resize (n, 0.0);
  // Get the local size
  VecGetLocalSize (v, &nlocal);

  // Gather all the local sizes in order to compute the
  // counts and displs for the Allgatherv
  int size, myrank;
  MPI_Comm_rank (comm, &myrank);
  MPI_Comm_size (comm, &size);
  vector < int >counts (size), displs (size);
  MPI_Allgather (&nlocal, 1, MPI_INT, &counts[0], 1, MPI_INT, comm);
  displs[0] = 0;
  for (int j = 1; j < size; j++)
    displs[j] = displs[j - 1] + counts[j - 1];

  // Get the internal values of the PETSc vector
  double *vp;
  VecGetArray (v, &vp);
  // Do the Allgatherv to the local vector
  MPI_Allgatherv (vp, nlocal, MPI_DOUBLE,
		  &values[0], &counts[0], &displs[0], MPI_DOUBLE, comm);
  // Restore the array
  VecRestoreArray (v, &vp);
}

//---:---<*>---:---<*>---:---<*>---:---<*>---:---<*>
int
h5petsc_vec_save (Vec x, const char *filename, const char *varname)
{
  vector < double >vx;
  vec_gather (PETSC_COMM_WORLD, x, vx);
  H5::H5File file (filename, H5F_ACC_TRUNC);
  hsize_t n = vx.size ();
  H5::DataSpace dataspace (1, &n);
  // Create the dataset.
  string svar (varname);
  H5::DataSet xdset =
    file.createDataSet (svar, H5::PredType::NATIVE_DOUBLE, dataspace);
  xdset.write (vx.data (), H5::PredType::NATIVE_DOUBLE);
  file.close ();
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "main"
int
main (int argc, char **argv)
{
  SNES snes;			/* nonlinear solver context */
  Vec x, r;			/* solution, residual vectors */
  Mat J;			/* Jacobian matrix */
  int its, size, rank;
  PetscErrorCode ierr;
  PetscInt T_o, T_f, I, i, j;
  PetscScalar coef;
  SnesCtx ctx;
  int N = 100;			//valor por default

  PetscInitialize (&argc, &argv, (char *) 0, help);	/*Inicializo PETSc */
  ierr = MPI_Comm_size (PETSC_COMM_WORLD, &size);
  CHKERRQ (ierr);
  ierr = MPI_Comm_rank (PETSC_COMM_WORLD, &rank);
  CHKERRQ (ierr);
   
    //Tmb puede usarse la siguiente sintaxis
//#######--------------------------#######
//  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
//  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
    //si fuera para un solo procesador e intentaramos correrlo con mpirun -np>1
//  PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "Example is only for sequential runs");
//#######--------------------------#######
    PetscPrintf (PETSC_COMM_WORLD, "Cantidad de procesos: %D \n", size);

  /*Tomo el valor de tamaño de elementos por lado de la grilla ingresado via command line */
  ierr = PetscOptionsGetInt (PETSC_NULL, "-N", &N, PETSC_NULL);
  CHKERRQ (ierr);

  /*Verifico que #filas=N+1 sea multiplo de #proc; caso contrario corrijo N */
  if ((N + 1) % size != 0)
    {
      N = N - (N % size) + size - 1;
      PetscPrintf (PETSC_COMM_WORLD,
		   "Se modifico N a su nuevo valor N=%D para que N+1 sea multiplo de #proc=%D \n",
		   N, size);
    }
  else
    {
      PetscPrintf (PETSC_COMM_WORLD,
		   "Se mantiene N original de valor N=%D \n", N);
    }

  ctx.N = N;
  ctx.k = 3e-3;
  ctx.c = 1;
  ctx.f0 = 0.0;
  ctx.h = 1.0 / N;
    PetscPrintf (PETSC_COMM_WORLD,
		"Parametros del problema: N=%D, k= %f, c= %f, h=%f \n", ctx.N,
		ctx.k, ctx.c, ctx.h);
  
    // Create nonlinear solver context
  ierr = SNESCreate (PETSC_COMM_WORLD, &snes);CHKERRQ (ierr);
  ierr = SNESSetType (snes, SNESLS);CHKERRQ (ierr);
  SNESSetFromOptions (snes);

  //Seteo parametros de SNES (tolerancias, max iter, etc)
  double abstol, rtol, stol;
  int maxit, maxf;
  SNESGetTolerances (snes, &abstol, &rtol, &stol, &maxit, &maxf);
  PetscPrintf (PETSC_COMM_WORLD,
	       "atol=%g, rtol=%g, stol=%g, maxit=%D, maxf=%D\n",
	       (double) abstol, (double) rtol, (double) stol, maxit, maxf);

  // Create matrix and vector data structures;
  // set corresponding routines

  // Create vectors for solution and nonlinear function
//  int m=(N+1)/size;
  ierr = VecCreate (PETSC_COMM_WORLD, &x);CHKERRQ (ierr);
  ierr = VecSetSizes (x, PETSC_DECIDE, (N + 1) * (N + 1));CHKERRQ (ierr);		//PETSC_DECIDE
  ierr = VecSetFromOptions (x);CHKERRQ (ierr);
  ierr = VecSetUp (x);CHKERRQ (ierr);

  //Usando al vector solucion x, creo el vector residuo r
  ierr = VecDuplicate (x, &r);CHKERRQ (ierr);

  //Determino que rango de filas son locales a cada proc
  ierr = VecGetOwnershipRange (x, &T_o, &T_f);CHKERRQ (ierr);

  //Seteo Condicion de borde
  for (I = T_o; I < T_f; I++)
    {
      i = I / (N + 1);
      j = I - i * (N + 1);
      double x_coord = i * ctx.h, y_coord = j * ctx.h;
      double phi = x_coord > 0.5;
      ierr = VecSetValue (x, I, phi, ADD_VALUES);CHKERRQ (ierr);
//    if (j<(N+1)/2){//i==0 || i==N || j==0 || j==N
//      coef=0;
//      ierr = VecSetValues(x,1,&I,&coef,INSERT_VALUES); CHKERRQ(ierr);
//    }
//    else{ coef = 1; //((I%(N+1))>((N+1)/4.0))
//      ierr = VecSetValues(x,1,&I,&coef,INSERT_VALUES); CHKERRQ(ierr);
//    }
    }
  ierr = VecAssemblyBegin (x);CHKERRQ (ierr);		//ensamblo el vector
  ierr = VecAssemblyEnd (x);CHKERRQ (ierr);

#if 1
  h5petsc_vec_save (x, "temp_inicial.h5", "u_i");
#endif

  ierr = MatCreate (PETSC_COMM_WORLD, &J);CHKERRQ (ierr);
  ierr = MatSetSizes (J, PETSC_DECIDE, PETSC_DECIDE, (N + 1) * (N + 1),
		 (N + 1) * (N + 1));CHKERRQ (ierr);
  ierr = MatSetFromOptions (J);CHKERRQ (ierr);
  ierr = MatSetUp (J);CHKERRQ (ierr);

  ierr = SNESSetFunction (snes, r, resfun, &ctx);CHKERRQ (ierr);
  ierr = SNESSetJacobian (snes, J, J, jacfun, &ctx);CHKERRQ (ierr);
   
//  MPI_Barrier(PETSC_COMM_WORLD);
    ierr = SNESSolve (snes, NULL, x);CHKERRQ (ierr);

#if 0
  // Doesn't work
  PetscObjectSetName ((PetscObject) x, "temp");
  PetscViewer viewer;
  ierr = PetscViewerHDF5Open (PETSC_COMM_WORLD, "./temp.h5",
			      FILE_MODE_WRITE, &viewer);CHKERRQ (ierr);
  ierr = VecView (x, viewer);CHKERRQ (ierr);
  ierr = PetscViewerDestroy (&viewer);CHKERRQ (ierr);
#elif 1
  // Use the function defined above
  h5petsc_vec_save (x, "temp.h5", "u");
#else
  // Just plain output to stdout
  ierr = VecView (x, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ (ierr);
#endif

  Vec f;
  ierr = SNESGetFunction (snes, &f, 0, 0);CHKERRQ (ierr);
  double rnorm;
  ierr = VecNorm (r, NORM_2, &rnorm);

  SNESGetLinearSolveIterations (snes, &its);
  ierr = PetscPrintf (PETSC_COMM_SELF,
		      "number of Newton iterations = "
		      "%d, norm res %g\n", its, rnorm);CHKERRQ (ierr);

  //Destruyo los objetos de PETSc creados y usados
  ierr = VecDestroy (&x);CHKERRQ (ierr);
  ierr = VecDestroy (&r);CHKERRQ (ierr);
  ierr = MatDestroy (&J);CHKERRQ (ierr);
  ierr = SNESDestroy (&snes);CHKERRQ (ierr);

  ierr = PetscFinalize ();CHKERRQ (ierr);
  return 0;
}

//---:---<*>---:---<*>---:---<*>---:---<*>---:---<*>
#undef __FUNCT__
#define __FUNCT__ "resfun"
int
resfun (SNES snes, Vec x, Vec f, void *data)
{
  //Casteo el void * a un puntero de tipo SnesCtx*
  SnesCtx & ctx = *(SnesCtx *) data;
  PetscErrorCode ierr;
  double h = ctx.h;
  PetscInt i, j, I, Ires_o, Ires_f, T_o, T_f;
  PetscScalar ff;

  ierr = VecGetOwnershipRange (f, &Ires_o, &Ires_f);
  ierr = VecGetOwnershipRange (x, &T_o, &T_f);

  vector < double >xx;
  vec_gather (PETSC_COMM_WORLD, x, xx);
  
    //Lleno el vector f
    for (I = Ires_o; I < Ires_f; I++)
    {
      i = I / (ctx.N + 1);
      j = I - i * (ctx.N + 1);
      double x_coord = i * ctx.h;
      if (i == 0 || i == ctx.N || j == 0 || j == ctx.N)
	{			//i==0 || i==ctx.N || j==0 || j==ctx.N
	  //Nodos de frontera
	  if (x_coord < 0.5)
	    {
	      ff = xx[I];	//no resto nada por la CB en esta zona del dominio (T=0 para x<0.5)
	      ierr = VecSetValues (f, 1, &I, &ff, INSERT_VALUES);CHKERRQ (ierr);
	    }
	  else
	    {
	      ff = xx[I] - 1;	//resto 1 por la CB en esta zona del dominio (T=1 para x>=0.5)
	      ierr = VecSetValues (f, 1, &I, &ff, INSERT_VALUES);CHKERRQ (ierr);
	    }
	}
      else
	{
	  //Nodos interiores
	  if (x_coord < 0.5)
	    {
	      double xxx = xx[I];	//no resto nada por la CB en esta zona del dominio (T=0 para x<0.5)
	      ff = ctx.f0 + ctx.c * xxx * (0.5 - xxx) * (1.0 - xxx);
	      ff +=
		ctx.k * (-xx[I + 1] - xx[I - (ctx.N + 1)] + 4 * xx[I] -
			 xx[I + (ctx.N + 1)] - xx[I - 1]) / (h * h);
	      ierr = VecSetValues (f, 1, &I, &ff, INSERT_VALUES);CHKERRQ (ierr);
	    }
	  else
	    {
	      double xxx = xx[I] - 1;	//resto 1 por la CB en esta zona del dominio (T=1 para x>=0.5)
	      ff = ctx.f0 + ctx.c * xxx * (0.5 - xxx) * (1.0 - xxx);
	      ff +=
		ctx.k * (-xx[I + 1] - xx[I - (ctx.N + 1)] + 4 * xx[I] -
			 xx[I + (ctx.N + 1)] - xx[I - 1]) / (h * h);
	      ierr = VecSetValues (f, 1, &I, &ff, INSERT_VALUES);CHKERRQ (ierr);
	    }
	}
    }

  // Ensamblo vector f
  ierr = VecAssemblyBegin (f);CHKERRQ (ierr);
  ierr = VecAssemblyEnd (f);CHKERRQ (ierr);

#if 0
  ierr = PetscPrintf (PETSC_COMM_SELF, "x:\n");
  ierr = VecView (x, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ (ierr);
  ierr = PetscPrintf (PETSC_COMM_SELF, "f:\n");
  ierr = VecView (f, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ (ierr);
  ierr = PetscPrintf (PETSC_COMM_SELF, "================\n");
#endif
  return 0;
}

//---:---<*>---:---<*>---:---<*>---:---<*>---:---<*>
#undef __FUNCT__
#define __FUNCT__ "jacfun"
int
jacfun (SNES snes, Vec x, Mat * jac, Mat * jac1,
	MatStructure * flag, void *data)
{
  //Casteo el void * a un puntero de tipo SnesCtx*
  SnesCtx & ctx = *(SnesCtx *) data;
  PetscErrorCode ierr;
  vector < double >xx;
  double A, coef, coefs[5];
  int cols[5], rank, size;
  PetscInt i, j, I, Ijac_o, Ijac_f, N, T_o, T_f;
  ierr = MPI_Comm_size (PETSC_COMM_WORLD, &size);CHKERRQ (ierr);
  ierr = MPI_Comm_rank (PETSC_COMM_WORLD, &rank);CHKERRQ (ierr);

  vec_gather (PETSC_COMM_WORLD, x, xx);
  ierr = MatZeroEntries (*jac);

  double h = ctx.h, h2 = h * h;

  //stencil 2D de 5 posiciones tipo von Neumann para el Laplaciano
  coefs[0] = -ctx.k * 1.0 / h2;
  coefs[1] = -ctx.k * 1.0 / h2;
  coefs[2] = +ctx.k * 4.0 / h2;
  coefs[3] = -ctx.k * 1.0 / h2;
  coefs[4] = -ctx.k * 1.0 / h2;

  ierr = MatGetOwnershipRange (*jac, &Ijac_o, &Ijac_f);CHKERRQ (ierr);
  ierr = VecGetOwnershipRange (x, &T_o, &T_f);CHKERRQ (ierr);

  N = ctx.N;
  for (I = Ijac_o; I < Ijac_f; I++)
    {
      i = I / (N + 1);
      j = I - i * (N + 1);
      //Nodos de frontera
      if (i == 0 || i == N || j == 0 || j == N)
	{
	  coef = 1;
	  ierr = MatSetValues (*jac, 1, &I, 1, &I, &coef, ADD_VALUES);
	  CHKERRQ (ierr);
	}
      else
	{			//Nodos interiores
	  double xxx = xx[I];
	  //primero sumo la parte reactiva no lineal (solo contribuye en la diagonal)
	  A =
	    ctx.c * ((0.5 - xxx) * (1.0 - xxx) - xxx * (1.0 - xxx) -
		     xxx * (0.5 - xxx));
	  ierr = MatSetValues (*jac, 1, &I, 1, &I, &A, ADD_VALUES);
	  
	  cols[0] = I - (N + 1);
	  cols[1] = I + 1;
	  cols[2] = I;
	  cols[3] = I - 1;
	  cols[4] = I + (N + 1);
	  //adiciono la parte del Laplaciano (parte diagonal y parte no diagonal)
	  ierr = MatSetValues (*jac, 1, &I, 5, cols, coefs, ADD_VALUES);
	  CHKERRQ (ierr);
	}
    }
  //Ensamblo la matriz
  ierr = MatAssemblyBegin (*jac, MAT_FINAL_ASSEMBLY);CHKERRQ (ierr);
  ierr = MatAssemblyEnd (*jac, MAT_FINAL_ASSEMBLY);CHKERRQ (ierr);

  return 0;
}
