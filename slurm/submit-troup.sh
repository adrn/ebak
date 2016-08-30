#!/bin/bash
#SBATCH -J troup           # job name
#SBATCH -o troup.o%j             # output file name (%j expands to jobID)
#SBATCH -e troup.e%j             # error file name (%j expands to jobID)
#SBATCH -n 128                   # total number of mpi tasks requested
#SBATCH -p normal          # queue (partition) -- normal, development, etc.
#SBATCH -A TG-AST150023         # Project ID
#SBATCH -t 12:00:00             # run time (hh:mm:ss) - 1.5 hours
#SBATCH --mail-user=amp2217@columbia.edu
#SBATCH --mail-type=begin       # email me when the job starts
#SBATCH --mail-type=end         # email me when the job finishes

export SAMPLER='kombine'
export NBURN=128 # Irrelevant for Kombine
export NSTEPS=4096
export NWALKERS=512

cd $WORK/projects/ebak/scripts/

source activate ebak

NTROUP=382
for ((i=0; i<NTROUP; i++)); do
    ibrun python mcmc-troup.py -v -s $SAMPLER -o --n-steps=$NSTEPS --n-burnin=$NBURN --n-walkers=$NWALKERS --index=$i --mpi
    wait
done
