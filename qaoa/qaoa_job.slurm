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
# Replace <project ID> with your Pawsey project ID (e.g. pawsey0001).
#SBATCH --account=<project ID>-q    # Project account for billing/allocation
#SBATCH --partition=quantum        # Use the quantum partition (GPU-enabled)
#SBATCH --gpus=1                   # Number of GPUs (Allocation packs)
#SBATCH --time=00:10:00            # Maximum walltime (HH:MM:SS)
#SBATCH --export=NONE              # Start with clean environment
#SBATCH --output=%x-%j.out         # Stdout file: %x=job-name, %j=job-id
#SBATCH --error=%x-%j.err          # Stderr file: captures error messages separately

# -----------------------------------------------------------------------------
# Environment Setup
# -----------------------------------------------------------------------------
# Clear any pre-loaded modules and load the Setonix-Q 2026.08 software stack
module purge                           # Remove all loaded modules
module load pawsey pawseytools         # Pawsey base modules
module use /software/setonix-q/2026.08/preview_modulefiles
module load pawseyenv/2026.08          # Pawsey 2026.08 software stack
module load PrgEnv-gnu                 # GNU programming environment
module load py-qiskit/2.3.0-cuda-gh200 # GPU-enabled Qiskit
module list                            # Print loaded modules

# Set OpenMP threads to 1 to avoid oversubscription
export OMP_NUM_THREADS=1

srun -N 1 -n 1 -c 72 --gpus=1 python qaoa.py
