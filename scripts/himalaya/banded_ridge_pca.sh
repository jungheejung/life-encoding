#!/bin/bash -l

#SBATCH --job-name=life-himalaya
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem-per-cpu=8gb
#SBATCH --time=01:00:00
#SBATCH -o ./log_pca/hac_%A_%a.o
#SBATCH -e ./log_pca/hac_%A_%a.e
#SBATCH --account=DBIC
#SBATCH --partition=standard
#SBATCH --array=1-2
# Vertices are split into chunks 0-39
###SBATCH --array=1-5760%50 


conda activate himalaya

### FIX THIS
MAINDIR=/dartfs/rc/lab/D/DBIC/DBIC/f0042x1/life-encoding/scripts/himalaya
echo "SLURMSARRAY: " ${SLURM_ARRAY_TASK_ID}

# Subtract one to get python indexing
NUMBERS=$(sed -n "1p" output_pca_pca-40_align-ha_common.txt)
IFS=',' read -ra NUMBER_ARRAY <<< "$NUMBERS"
ID=$((NUMBER_ARRAY[${SLURM_ARRAY_TASK_ID}]-1))
echo ${ID}

# Use slurm_array.sh file to set parameters
ARRAY_FILE=${MAINDIR}/slurm_array.txt
echo ${ARRAY_FILE}
INFILE=`awk -F "," -v RS="\n" "NR==${ID}" ${ARRAY_FILE}`
subject=$(echo $INFILE | cut -f1 -d,)
hemisphere=$(echo $INFILE | cut -f2 -d,)
test_run=$(echo $INFILE | cut -f3 -d,)
roi=$(echo $INFILE | cut -f4 -d,)

# Set command line arguments for banded_ridge.py
alignment="ha_common" # ws, aa, ha_common, ha_test
features="bg actions agents"

echo ${subject} ${hemisphere} ${test_run} ${roi}
python ${MAINDIR}/banded_ridge_pca.py -a ${alignment} --hemisphere ${hemisphere} --test-run ${test_run} -s ${subject} -f ${features} --roi ${roi}
