#!/bin/bash -l

#SBATCH --job-name=life_tikreg
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mem-per-cpu=8gb
#SBATCH --time=12:00:00
#SBATCH -o ./log_roi/ROI_tikreg_vt_%A_%a.o
#SBATCH -e ./log_roi/ROI_tikreg_vt_%A_%a.e
#SBATCH --account=DBIC
#SBATCH --partition=standard
#SBATCH --array=4-7

module load python/2.7-Anaconda
source /optnfs/common/miniconda3/etc/profile.d/conda.sh
conda activate /dartfs-hpc/rc/home/1/f0042x1/.conda/envs/haxby_mvpc

MAINDIR=/dartfs/rc/lab/D/DBIC/DBIC/f0042x1/life-encoding/scripts/tikreg-pca

echo "SLURMSARRAY: " ${SLURM_ARRAY_TASK_ID}
IND=$((SLURM_ARRAY_TASK_ID-1))
#INFILE=`awk "NR==${IND}" ./roi_ips.txt`
INFILE=`awk -v RS='\r\n' "NR==${SLURM_ARRAY_TASK_ID}" roi_vt.txt`
echo $INFILE
SUB=$(echo $INFILE | cut -f1 -d,)
RUN=$(echo $INFILE | cut -f2 -d,)
HEMI=$(echo $INFILE | cut -f3 -d,)
NODE=$(echo $INFILE | cut -f4 -d,)

MODEL="visual"
ALIGN="ws"
STIM1="bg"
STIM2="actions"
STIM3="agents"

hostname -s
python ./PCATIKREG_05ROI_fullrange_10000.py ${MODEL} ${ALIGN} ${STIM1} ${STIM2} ${STIM3} ${RUN} ${HEMI} ${SUB} ${NODE}
