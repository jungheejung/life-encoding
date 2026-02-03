#!/bin/bash
#SBATCH --job-name=spacetop_resample
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/resample_%A_%a.out
#SBATCH --array=0-11  # If you have 12 subjects

module load connectome-workbench/2.0.1 
module load python/2.7-Anaconda
source /optnfs/common/miniconda3/etc/profile.d/conda.sh
conda activate himalaya

# 1. Create a list of your subjects (matching your directory names)
SUBJECTS=("sub-0051" "sub-0052" "sub-0053" "sub-0054" "sub-0055" "sub-0056" "sub-0057" "sub-0058" "sub-0059" "sub-0060" "sub-0061" "sub-0062")

# 2. Get the subject for THIS specific array task
CURRENT_SUB=${SUBJECTS[$SLURM_ARRAY_TASK_ID]}

# 3. Load your environment (conda, workbench, etc.)

# 4. Run the python script with the subject parameter
python preproc_refactor.py --sub $CURRENT_SUB
