#!/bin/bash -l

#SBATCH --job-name=moten
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem-per-cpu=8gb
#SBATCH --time=04:00:00
#SBATCH -o ./log/moten_%A_%a.o
#SBATCH -e ./log/moten_%A_%a.e
#SBATCH --account=DBIC
#SBATCH --partition=standard
#SBATCH --array=1-5


conda activate pymoten

MAINDIR=/dartfs/rc/lab/D/DBIC/DBIC/f0042x1/life-encoding/scripts/motion_energy
python ${MAINDIR}/moten_extract.py
