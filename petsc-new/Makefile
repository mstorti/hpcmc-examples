## PROG := ex1b
PROG := snes2
default: $(PROG).bin

PUB:=/share/storage/pub/mstorti
PETSC_DIR := $(PUB)/petsc-3.23.2
PETSC_ARCH := linux-gnu-O
# include ${PETSC_DIR}/conf/variables
# include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

## COPTFLAGS := -O2 -funroll-loops
COPTFLAGS := -O0 -g3 -ggdb3 
## LDPRE := LD_PRELOAD=$(HOME)/SOFT/hdf5-1814/lib/libhdf5.so
LDPRE := LD_LIBRARY_PATH=$(HOME)/SOFT/hdf5-1814/lib
EIGEN := -I/usr/include/eigen3

$(PROG).bin: $(PROG).cpp
	mpicxx $(COPTFLAGS) $(PETSC_CC_INCLUDES)			\
		-o $@ $^ -L$(HDF5)/lib  -Wl,-rpath,$(HDF5)/lib		\
		-lhdf5 -lhdf5_cpp $(PETSC_LIB)  

run:
	$(LDPRE) mpiexec -n 1 snes2.bin -n 200

#w Describe targets
what:
	@makewhat.pl -s Makefile

#w Remove all bins
lclean:
	shopt -s nullglob ; rm -f slurm* *~ *.bin
