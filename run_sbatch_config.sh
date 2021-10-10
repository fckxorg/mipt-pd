#!/bin/bash
#
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --partition=RT
#SBATCH --job-name="MIPT-PD homework"
#SBATCH --comment="Run mpi from config"
#SBATCH --output=output.txt
#SBATCH --error=error.txt
mpiexec ./a.out $1
