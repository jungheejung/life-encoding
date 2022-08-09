#!/bin/bash -l

#SBATCH --job-name=life-himalaya
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=8gb
#SBATCH --time=1:00:00
#SBATCH -o ./log_roi/life_himalaya_%A_%a.o
#SBATCH -e ./log_roi/life_himalaya_%A_%a.e
#SBATCH --account=DBIC
#SBATCH --partition=standard

# Vertices are split into chunks 0-39
#SBATCH --array=1-3 
#-40

conda activate himalaya

### FIX THIS
MAINDIR=/dartfs/rc/lab/D/DBIC/DBIC/f0042x1/life-encoding/scripts/tikreg-pca
MAINDIR="/dartfs/rc/lab/D/DBIC/DBIC/f0042x1/life-encoding/scripts/himalaya"
echo "SLURMSARRAY: " ${SLURM_ARRAY_TASK_ID}

# Subtract one to get python indexing
ID=$((SLURM_ARRAY_TASK_ID-1))

# Use slurm_array.sh file to set parameters
ARRAY_FILE=${MAIN_DIR}/slurm_array.txt
INFILE=`awk -F "," -v RS="\n" "NR==${ID}" ${ARRAY_FILE}`
subject=$(echo $INFILE | cut -f1 -d,)
hemisphere=$(echo $INFILE | cut -f2 -d,)
test_run=$(echo $INFILE | cut -f3 -d,)
roi=$(echo $INFILE | cut -f4 -d,)

# Set command line arguments for banded_ridge.py
alignment="ha_common" # ws, aa, ha_common, ha_test
features="bg actions agents"

python ${MAINDIR}/banded_ridge.py -a ${alignment} --hemisphere ${hemisphere} --test-run ${test_run} -s ${test_subject} -f ${features} --roi ${roi}
