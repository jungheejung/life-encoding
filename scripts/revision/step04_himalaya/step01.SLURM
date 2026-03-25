#!/bin/bash

# --- Paths ---
SCRIPT_DIR="/vast/labs/DBIC/datasets/Life/life-encoding"
MAINDIR="/dartfs/rc/lab/H/HaxbyLab/heejung"
DATADIR="/dartfs/rc/lab/H/HaxbyLab/heejung/data_spacetoptrim"
PARAM_FILE="./params_test.txt"

# --- Parameters ---
subjects=($(find "$DATADIR" -maxdepth 1 -type d -name "sub-*" -exec basename {} \; | sort))
hemispheres=("lh" "rh")
test_runs=(1 2 3 4)
alignment="ws"
features="scenes actions agents objects moten"

# --- 1. Generate Parameter File (Limit to 10) ---
> $PARAM_FILE  # Clear/Create file
count=0
MAX_TEST=10

for sub in "${subjects[@]}"; do
    for hemi in "${hemispheres[@]}"; do
        for run in "${test_runs[@]}"; do
            if [ "$count" -lt "$MAX_TEST" ]; then
                echo "$sub,$hemi,$run" >> $PARAM_FILE
                ((count++))
            fi
        done
    done
done

echo "Generated $PARAM_FILE with $count jobs."

# --- 2. Submit the Job Array ---
# We use 1-10 because Slurm arrays are 1-indexed usually
sbatch <<EOT
#!/bin/bash -l
#SBATCH --job-name=himalaya_array
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem-per-cpu=8gb
#SBATCH --time=05:30:00
#SBATCH --account=DBIC
#SBATCH --partition=standard
#SBATCH --array=1-$MAX_TEST
#SBATCH -o ./log_glove/array_%A_%a.o
#SBATCH -e ./log_glove/array_%A_%a.e

conda activate himalaya

# Extract parameters for THIS specific task
LINE=\$(sed -n "\${SLURM_ARRAY_TASK_ID}p" $PARAM_FILE)
IFS=',' read -ra PARAMS <<< "\$LINE"

SUB=\${PARAMS[0]}
HEMI=\${PARAMS[1]}
RUN=\${PARAMS[2]}

echo "Running Task \$SLURM_ARRAY_TASK_ID: Subject: \$SUB, Hemi: \$HEMI, Run: \$RUN"

python ${SCRIPT_DIR}/scripts/revision/step04_himalaya/step01_banded_ridge_moten_0tr_glove.py \
    --maindir ${MAINDIR} \
    -a ${alignment} \
    --hemisphere \$HEMI \
    --test-run \$RUN \
    -s \$SUB \
    -f ${features}
EOT

echo "Array job submitted. You can delete $PARAM_FILE after jobs finish."