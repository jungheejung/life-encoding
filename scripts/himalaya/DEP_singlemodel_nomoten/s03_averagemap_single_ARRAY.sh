#!/bin/bash -l

#SBATCH --job-name=life-himalaya
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem-per-cpu=8gb
#SBATCH --time=01:00:00
#SBATCH -o ./output/moten40_%A_%a.o
#SBATCH -e ./output/moten40_%A_%a.e
#SBATCH --account=DBIC
#SBATCH --partition=standard
#SBATCH --array=1-20
conda activate himalaya

MAINDIR=/dartfs/rc/lab/D/DBIC/DBIC/f0042x1/life-encoding/scripts/himalaya
echo "SLURMSARRAY: " ${SLURM_ARRAY_TASK_ID}

# Subtract one to get python indexing
ID=$((SLURM_ARRAY_TASK_ID-1))
ALIGN="ha_common"
PCA=40
FEATURES="agents"
python ${MAINDIR}/singlemodel/average_maps_single.py \
--index ${ID} \
--align ${ALIGN} \
--pca ${PCA} \
-f ${FEATURES} \
