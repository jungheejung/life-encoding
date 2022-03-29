#!/bin/bash -l
#SBATCH --job-name=life
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=24:00:00
#SBATCH -o ./log/%A_%a.o
#SBATCH -e ./log/%A_%a.e
#SBATCH --account=DBIC
#SBATCH --partition=standard
#SBATCH --array=1

module load python/2.7-Anaconda
conda activate haxby_mvpc

python2 apply_mappers.py 
