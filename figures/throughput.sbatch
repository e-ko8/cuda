#!/bin/bash
#SBATCH --job-name=throughput_<NTASKS> # Job name
#SBATCH --ntasks=128                  # Number of MPI tasks (i.e. processes)
#SBATCH --cpus-per-task=1             # Number of cores per MPI task 
#SBATCH --distribution=cyclic:cyclic  # Distribute tasks cyclically first among nodes and then among sockets within a node
#SBATCH --time=00:05:00               # Wall time limit (days-hrs:min:sec)
#SBATCH --output=throughput_%j.log     # Path to the standard output and error files relative to the working directory

echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo ""
echo "Number of Nodes Allocated      = $SLURM_JOB_NUM_NODES"
echo "Number of Tasks Allocated      = $SLURM_NTASKS"
echo "Number of Cores/Task Allocated = $SLURM_CPUS_PER_TASK"

srun --mpi=pmix_v3 ./throughput
