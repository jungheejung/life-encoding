#!/bin/bash -l
#PBS -N life_encode_submit_py
#PBS -q default
#PBS -l nodes=1:ppn=16
#PBS -l walltime=06:00:00
#PBS -m bea
#PBS -A DBIC

cd $PBS_O_WORKDIR

module load python/2.7-Anaconda
source /optnfs/common/miniconda3/etc/profile.d/conda.sh
conda activate haxby_mvpc

MODEL="visual"
ALIGN="ws"
STIM1="bg"
STIM2="actions"
SUB=${1}
RUN=${2}
hemis=("lh" "rh")
MAINDIR=/dartfs/rc/lab/D/DBIC/DBIC/f0042x1/life-encoding/scripts/s03_banded-ridge


for HEMI in ${hemis[*]}; do
CMD="python ${MAINDIR}/c03_banded_ridge.py  ${MODEL} ${ALIGN} ${STIM1} ${STIM2} ${RUN} ${HEMI} ${SUB}"
echo "Running $CMD"
exec $CMD
done
