#!/bin/bash -l

#SBATCH --job-name=life-himalaya
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem-per-cpu=8gb
#SBATCH --time=6:00:00
#SBATCH -o ./log_roi/life_himalaya_%A_%a.o
#SBATCH -e ./log_roi/life_himalaya_%A_%a.e
#SBATCH --account=DBIC
#SBATCH --partition=standard

# Vertices are split into chunks 0-39
#SBATCH --array=1 
#-40

conda activate himalaya

### FIX THIS
MAINDIR=/dartfs/rc/lab/D/DBIC/DBIC/f0042x1/life-encoding/scripts/tikreg-pca

echo "SLURMSARRAY: " ${SLURM_ARRAY_TASK_ID}

# Subtract one to get python indexing
ID=$((SLURM_ARRAY_TASK_ID-1))

# Set command line arguments for banded_ridge.py
alignment="ws" # ws, aa, ha_common, ha_test
hemisphere="lh" # lh, rh
test_run="1" # 1, 2, 3, 4
test_subject="sub-rid000005"
features="bg actions agents"
roi=$ID

hostname -s
python banded_ridge.py -a $alignment --hemisphere $hemisphere -r $test_run \
                       -s $test_subject -f $features -r $roi
