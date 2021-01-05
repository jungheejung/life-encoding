#!/bin/bash -l
#PBS -A DBIC
#PBS -q default
#PBS -N isc
#PBS -M heejung.jung.gr@dartmouth.edu
#PBS -m ea
#PBS -l nodes=1:ppn=10
#PBS -l walltime=12:00:00

cd $PBS_O_WORKDIR

module load python/2.7-Anaconda
source /optnfs/common/miniconda3/etc/profile.d/conda.sh
conda activate haxby_mvpc
python /dartfs/rc/lab/D/DBIC/DBIC/f0042x1/life-encoding/scripts/s03_banded_cara-alpha_loro/ISC02_leave_one_subject_out.py
