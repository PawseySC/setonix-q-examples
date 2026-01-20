#!/bin/bash --login
# =============================================================================
# SLURM Job Script for QAOA on Setonix "quantum" Partition
# =============================================================================
# This script submits a GPU-accelerated quantum simulation job to the Setonix
# supercomputer at Pawsey. Submit with: sbatch qaoa_job.slurm
# =============================================================================

# -----------------------------------------------------------------------------
# SLURM Directives (Job Configuration)
# -----------------------------------------------------------------------------
#SBATCH --job-name=qaoa_job        # Name shown in queue (squeue command)
#SBATCH --account=pawsey0001-q     # Project account for billing/allocation
#SBATCH --partition=quantum        # Use the quantum partition (GPU-enabled)
#SBATCH --nodes=1                  # Number of compute nodes
#SBATCH --gpus=1                   # Number of GPUs (Allocation packs)
#SBATCH --time=00:10:00            # Maximum walltime (HH:MM:SS) - job killed if exceeded
#SBATCH --export=NONE              # Start with clean environment (recommended)
#SBATCH --output=%x-%j.out         # Stdout file: %x=job-name, %j=job-id
#SBATCH --error=%x-%j.err          # Stderr file: captures error messages separately

# -----------------------------------------------------------------------------
# Environment Setup
# -----------------------------------------------------------------------------
# Clear any pre-loaded modules and load required software stack
module purge                           # Remove all loaded modules
module load pawsey pawseytools pawseyenv/2026.01  # Pawsey base environment
module load PrgEnv-gnu-nvidia          # GNU compilers + NVIDIA CUDA support
module load py-qiskit/2.3.0            # Qiskit quantum computing framework
module list                            # Print loaded modules (useful for debugging)

# Set OpenMP threads to 1 to avoid oversubscription
export OMP_NUM_THREADS=1
# -N 1: use 1 node, -n 1: run 1 task, -c 1: use 1 CPU core per task
srun -N 1 -n 1 -c 1 --gpus=1 python qaoa.py 
