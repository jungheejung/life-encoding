#!/bin/bash -l
#SBATCH --job-name=moten
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=12gb
#SBATCH --time=24:00:00
#SBATCH -o ./log/moten_%A_%a.o
#SBATCH -e ./log/moten_%A_%a.e
#SBATCH --account=DBIC
#SBATCH --partition=standard
#SBATCH --array=1-4


conda activate pymoten
ID=$((SLURM_ARRAY_TASK_ID))
MAINDIR="/dartfs/rc/lab/D/DBIC/DBIC/f0042x1/life-encoding/scripts/motion_energy"
python ${MAINDIR}/step01_moten_extract_batch.py --run ${ID}
