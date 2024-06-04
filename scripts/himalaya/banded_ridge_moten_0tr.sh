#!/bin/bash -l

#SBATCH --job-name=life-himalaya
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem-per-cpu=8gb
#SBATCH --time=05:30:00
#SBATCH -o ./log_0tr/hac_%A_%a.o
#SBATCH -e ./log_0tr/hac_%A_%a.e
#SBATCH --account=DBIC
#SBATCH --partition=standard
#SBATCH --array=966,5237,5244,5246,5247,5751,5752,5753,5754,5755,5756,5757,5758,5759,5760,5761
# Vertices are split into chunks 0-39


conda activate himalaya

### FIX THIS
MAINDIR=/dartfs/rc/lab/D/DBIC/DBIC/f0042x1/life-encoding/scripts/himalaya
echo "SLURMSARRAY: " ${SLURM_ARRAY_TASK_ID}

# Subtract one to get python indexing
ID=$((SLURM_ARRAY_TASK_ID-1))
NUMBERS=$(sed -n "1p" ${MAINDIR}/slurm_array.txt)
IFS=',' read -ra NUMBER_ARRAY <<< "$NUMBERS"
#ID=$((NUMBER_ARRAY[${SLURM_ARRAY_TASK_ID}]-1))
echo ${ID}
# Use slurm_array.sh file to set parameters
ARRAY_FILE="${MAINDIR}/slurm_array.txt"
echo ${ARRAY_FILE}
INFILE=`awk -F "," -v RS="\n" "NR==${ID}" ${ARRAY_FILE}`
subject=$(echo $INFILE | cut -f1 -d,)
hemisphere=$(echo $INFILE | cut -f2 -d,)
test_run=$(echo $INFILE | cut -f3 -d,)
roi=$(echo $INFILE | cut -f4 -d,)

# Set command line arguments for banded_ridge.py
alignment="ha_common" # ws, aa, ha_common, ha_test
features="bg actions agents moten"

echo ${subject} ${hemisphere} ${test_run} ${roi}
python ${MAINDIR}/banded_ridge_moten_0tr.py -a ${alignment} --hemisphere ${hemisphere} --test-run ${test_run} -s ${subject} -f ${features} --roi ${roi}
