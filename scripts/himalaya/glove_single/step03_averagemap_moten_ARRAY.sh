#!/bin/bash -l

#SBATCH --job-name=life-himalaya
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem-per-cpu=8gb
#SBATCH --time=01:00:00
#SBATCH -o ./recombine/moten40_%A_%a.o
#SBATCH -e ./recombine/moten40_%A_%a.e
#SBATCH --account=DBIC
#SBATCH --partition=standard
#SBATCH --array=1-20%5 


conda activate himalaya

MAINDIR=/dartfs/rc/lab/D/DBIC/DBIC/f0042x1/life-encoding/scripts/himalaya/glove_single
echo "SLURMSARRAY: " ${SLURM_ARRAY_TASK_ID}

# Subtract one to get python indexing
ID=$((SLURM_ARRAY_TASK_ID-1))
ALIGN="ha_common"
#"/moten/ha_common_pca-40" #aa ws
PCA=40
ANALYSIS="glove_single" # "0tr" # "pca"
FEATURES="bg moten" # "actions moten" "agents moten" "bg moten"

python ${MAINDIR}/step03_average_maps_moten.py \
--slurm-id ${ID} \
--align ${ALIGN} \
--analysis ${ANALYSIS} \
--features ${FEATURES} \
--pca ${PCA}
