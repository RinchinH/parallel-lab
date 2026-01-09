
#!/bin/bash
set -e

make

mpirun -np 4 ./jacobi_mpi
