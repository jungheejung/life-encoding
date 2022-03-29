#!/bin/bash -l

#SBATCH --job-name=himalaya
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mem-per-cpu=8gb
#SBATCH --time=12:00:00
#SBATCH -o ./log/%A_%a.o
#SBATCH -e ./log/%A_%a.e
#SBATCH --account=DBIC
#SBATCH --partition=standard
#SBATCH --array=1

source /optnfs/common/miniconda3/etc/profile.d/conda.sh
conda activate himalaya

MAINDIR=/dartfs/rc/lab/D/DBIC/DBIC/f0042x1/life-encoding/scripts/himalaya

echo "SLURMSARRAY: " ${SLURM_ARRAY_TASK_ID}

python himalaya_test_array.py --alignment ws --hemisphere lh --test-run 1 --test-subject sub-rid000005 --roi vt --features bg actions agents
