#!/bin/bash -l

#SBATCH --job-name=life-himalaya
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem-per-cpu=8gb
#SBATCH --time=01:00:00
#SBATCH -o ./avg_glove/moten40_%A_%a.o
#SBATCH -e ./avg_glove/moten40_%A_%a.e
#SBATCH --account=DBIC
#SBATCH --partition=standard
#SBATCH --array=1-20%5 


conda activate himalaya

MAINDIR=/dartfs/rc/lab/D/DBIC/DBIC/f0042x1/life-encoding/scripts/himalaya/glove
echo "SLURMSARRAY: " ${SLURM_ARRAY_TASK_ID}

# Subtract one to get python indexing
ID=$((SLURM_ARRAY_TASK_ID-1))
ALIGN="ha_common"
#"/moten/ha_common_pca-40" #aa ws
PCA=40
ANALYSIS="glove" # "0tr" # "pca"
python ${MAINDIR}/step03_average_maps_moten.py ${ID} ${ALIGN} ${PCA} ${ANALYSIS}
