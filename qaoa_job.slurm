#!/bin/bash --login
#SBATCH --job-name=qaoa_job
#SBATCH --account=pawsey0001-q
#SBATCH --partition=quantum
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:10:00
#SBATCH --export=NONE
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

module purge
module load pawsey pawseytools pawseyenv/2026.01
module load PrgEnv-gnu-nvidia
module load py-qiskit/2.3.0
module list

export OMP_NUM_THREADS=1

srun -N 1 -n 1 -c 1 --gpus=1 python qaoa.py
