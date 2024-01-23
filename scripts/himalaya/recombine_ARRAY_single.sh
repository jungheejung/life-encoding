#!/bin/bash -l

#SBATCH --job-name=life-himalaya
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --mem-per-cpu=6gb
#SBATCH --time=05:00:00
#SBATCH -o ./output/recombine_ws_%A_%a.o
#SBATCH -e ./output/recombine_ws_%A_%a.e
#SBATCH --account=DBIC
#SBATCH --partition=standard
#SBATCH --array=1-50

conda activate himalaya

MAINDIR=/dartfs/rc/lab/D/DBIC/DBIC/f0042x1/life-encoding/scripts/himalaya
echo "SLURMSARRAY: " ${SLURM_ARRAY_TASK_ID}

# Subtract one to get python indexing
ID=$((SLURM_ARRAY_TASK_ID-1))
ALIGN="ha_common" # "ws" "ha_common"
ANALYSIS="single"  # 'moten', 'base', 'pca'
features="bg moten" # "actions moten" "agents moten"
PC=40

python ${MAINDIR}/recombine_vertices_single.py \
--slurm-id ${ID} \
--align ${ALIGN} \
--analysis ${ANALYSIS} \
-f ${features} \
--pca ${PC}
