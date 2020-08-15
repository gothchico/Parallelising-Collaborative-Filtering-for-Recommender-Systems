CC=mpiCC
CFLAGS=-std=c++11

matFact_mpi: matFact_mpi.c
	$(CC) -g -o matFact_mpi matFact_mpi.c $(CFLAGS)
