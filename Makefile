default: snes2.bin


ifeq ($(HOSTNAME),urubu)
PETSC_DIR := $(HOME)/SOFT/petsc-3.2-p7
else
PETSC_DIR := $(HOME)/PETSC/petsc-3.2-p7
endif
PETSC_ARCH := linux-gnu-O
include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules

COPTFLAGS := -O2 -funroll-loops
LDPRE := LD_PRELOAD=$(HOME)/SOFT/hdf5-1814/lib/libhdf5.so

ex1.bin: ex1.cpp
	mpicxx $(COPTFLAGS) $(PETSC_CC_INCLUDES) -o $@ $^ $(PETSC_LIB)

ex23.bin: ex23.cpp
	mpicxx $(COPTFLAGS) $(PETSC_CC_INCLUDES) -o $@ $^ $(PETSC_LIB)

snes3.bin: snes3.cpp
	mpicxx $(COPTFLAGS) $(PETSC_CC_INCLUDES) -o $@ $^ $(PETSC_LIB)

poisson.bin: poisson.cpp
	mpicxx $(COPTFLAGS) $(PETSC_CC_INCLUDES) -o $@ $^ $(PETSC_LIB)

snes2.bin: snes2.cpp
	mpicxx $(COPTFLAGS) $(PETSC_CC_INCLUDES) -o $@ $^ $(PETSC_LIB)

lclean:
	shopt -s nullglob ; rm slurm* *~ *.bin

run:
	$(LDPRE) mpiexec -n 1 snes2.bin -n 200
