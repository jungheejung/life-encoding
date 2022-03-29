#!/bin/bash -l
#PBS -N life_encode_cara_alpha
#PBS -q default
#PBS -l nodes=1:ppn=16
#PBS -l walltime=24:00:00
#PBS -m bea
#PBS -A DBIC

cd $PBS_O_WORKDIR

module load python/2.7-Anaconda
source /optnfs/common/miniconda3/etc/profile.d/conda.sh
conda activate haxby_mvpc

MODEL="visual"
ALIGN=${1}
STIM="all"
RUN=${2}
hemis=("lh" "rh")
MAINDIR=/dartfs/rc/lab/D/DBIC/DBIC/f0042x1/life-encoding/scripts/s03_ridge-regression

for HEMI in ${hemis[*]}; do
python ${MAINDIR}/c03_ridge_regression.py  ${MODEL} ${ALIGN} ${STIM} ${RUN} ${HEMI}
echo "Running $CMD"
done
