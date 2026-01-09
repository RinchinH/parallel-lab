#!/bin/bash
#SBATCH --job-name=lab3_cuda
#SBATCH --time=00:20:00
#SBATCH --partition=gpuserv
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --output=lab3_cuda_%j.out
#SBATCH --error=lab3_cuda_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "PWD=$(pwd)"
echo "HOST=$(hostname)"
echo "JOBID=$SLURM_JOBID"

module purge
module load nvidia/cuda

echo "=== module list ==="
module list || true

echo "=== nvcc ==="
which nvcc
nvcc --version

echo "=== nvidia-smi ==="
nvidia-smi

bash scripts/run_cuda.sh 1000000 1e-8 10000
