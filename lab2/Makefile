
CXX = mpicxx
CXXFLAGS = -O3 -std=c++17

all: jacobi_mpi

jacobi_mpi: jacobi_mpi.cpp
    $(CXX) $(CXXFLAGS) -o $@ $<

run:
    mpirun -np 4 ./jacobi_mpi

clean:
    rm -f jacobi_mpi
