default: snes2.bin

ifeq ($(HOSTNAME),urubu)
PETSC_DIR := $(HOME)/SOFT/petsc-3.2-p7
else
PETSC_DIR := $(HOME)/PETSC/petsc-3.2-p7
## PETSC_DIR := $(HOME)/PETSC/petsc-3.7.4
HDF5 := $(HOME)/SOFT/hdf5-1814
endif
PETSC_ARCH := linux-gnu-O
include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules

COPTFLAGS := -O2 -funroll-loops
## LDPRE := LD_PRELOAD=$(HOME)/SOFT/hdf5-1814/lib/libhdf5.so
LDPRE := LD_LIBRARY_PATH=$(HOME)/SOFT/hdf5-1814/lib
EIGEN := -I/usr/include/eigen3

ex1.bin: ex1.cpp
	mpicxx $(COPTFLAGS) $(PETSC_CC_INCLUDES) -o $@ $^ $(PETSC_LIB)

ex23.bin: ex23.cpp
	mpicxx $(COPTFLAGS) $(PETSC_CC_INCLUDES) -o $@ $^ $(PETSC_LIB)

snes3.bin: snes3.cpp
	mpicxx $(COPTFLAGS) $(PETSC_CC_INCLUDES) -o $@ $^ $(PETSC_LIB)

poisson.bin: poisson.cpp
	mpicxx $(COPTFLAGS) $(PETSC_CC_INCLUDES) -o $@ $^ $(PETSC_LIB)

snes2.bin: snes2.cpp
	mpicxx $(COPTFLAGS) $(PETSC_CC_INCLUDES)		\
		-o $@ $^ -L$(HDF5)/lib  -Wl,-rpath,$(HDF5)/lib		\
		-lhdf5 -lhdf5_cpp -lhdf5_fortran  $(PETSC_LIB)  

snes2v.bin: snes2v.cpp
	mpicxx $(COPTFLAGS) $(PETSC_CC_INCLUDES)		\
		-o $@ $^ -L$(HDF5)/lib  -Wl,-rpath,$(HDF5)/lib		\
		-lhdf5 -lhdf5_cpp -lhdf5_fortran  $(PETSC_LIB)  

hornere.bin: hornere.cpp
	g++ -fopenmp -o $@ $^ $(EIGEN)

lclean:
	shopt -s nullglob ; rm slurm* *~ *.bin

run:
	$(LDPRE) mpiexec -n 1 snes2.bin -n 200
