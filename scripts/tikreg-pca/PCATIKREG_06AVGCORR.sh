#!/bin/bash -l

#SBATCH --job-name=life_tikreg
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem-per-cpu=8gb
#SBATCH --time=02:00:00
#SBATCH -o ./log_avg/ROI_tikreg_vt_%A_%a.o
#SBATCH -e ./log_avg/ROI_tikreg_vt_%A_%a.e
#SBATCH --account=DBIC
#SBATCH --partition=preemptable
#SBATCH --array=1

module load python/2.7-Anaconda
source /optnfs/common/miniconda3/etc/profile.d/conda.sh
conda activate haxby_mvpc

MAINDIR=/dartfs/rc/lab/D/DBIC/DBIC/f0042x1/life-encoding/scripts/tikreg-pca

echo "SLURMSARRAY: " ${SLURM_ARRAY_TASK_ID}
IND=$((SLURM_ARRAY_TASK_ID-1))
INFILE=`awk -v RS='\r\n' "NR==${SLURM_ARRAY_TASK_ID}" corr.txt`
echo $INFILE
FEATURE=$(echo $INFILE | cut -f1 -d,)
HEMI=$(echo $INFILE | cut -f2 -d,)
ROI=$(echo $INFILE | cut -f3 -d,)


hostname -s
python ./PCATIKREG_06avgcorr.py ${FEATURE} ${HEMI} ${ROI}
