#!/bin/bash -l
#PBS -A DBIC
#PBS -q default
#PBS -N br_avg
#PBS -M heejung.jung.gr@dartmouth.edu
#PBS -m ea
#PBS -l nodes=1:ppn=3
#PBS -l walltime=02:00:00

cd $PBS_O_WORKDIR

module load python/2.7-Anaconda
source /optnfs/common/miniconda3/etc/profile.d/conda.sh
conda activate haxby_mvpc
python /dartfs/rc/lab/D/DBIC/DBIC/f0042x1/life-encoding/scripts/s03_banded_cara-alpha_loro/BANDEDRIDGE03_average.py
