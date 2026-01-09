#!/bin/bash
#SBATCH --job-name=lab1_openmp
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --partition=cpu
#SBATCH --output=lab1_openmp_%j.out

module purge
# module load gcc/13.2

make -C lab1/openmp

bash lab1/scripts/run_openmp.sh 8000 1e-8 10000 fast
