#!/bin/bash

# NTROUP=382
NTROUP=2
for ((i=0; i<NTROUP; i++)); do
    export INDEX=$i
    sbatch --export=ALL _submit-one.sh
done
