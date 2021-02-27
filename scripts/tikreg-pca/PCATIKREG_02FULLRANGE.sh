#!/bin/bash -l
#PBS -N fe1aja
#PBS -q default
#PBS -l nodes=1:ppn=16
#PBS -l walltime=05:00:00
#PBS -A DBIC
#PBS -t 1-410
#PBS -o /log_fullrange/%o
#PBS -e /log_fullrange/%e

cd $PBS_O_WORKDIR

module load python/2.7-Anaconda
source /optnfs/common/miniconda3/etc/profile.d/conda.sh
conda activate haxby_mvpc

MAINDIR=/dartfs/rc/lab/D/DBIC/DBIC/f0042x1/life-encoding/scripts/s03_banded_cara-alpha_loro

echo ${PBS_ARRAYID}
echo "PBSARRAY"
# INFILE=`awk "NR==${PBS_ARRAYID}" ${MAINDIR}/canonical_sublist.txt`
INFILE=`awk "NR==${PBS_ARRAYID}" ${MAINDIR}/canonical_sublist.txt`
# SUB=$(echo $INFILE | cut -f1 -d,)
# RUN=$(echo $INFILE | cut -f2 -d,)
# HEMI=$(echo $INFILE | cut -f3 -d,)
# NODE=$(echo $INFILE | cut -f4 -d,)
SUB=$1
RUN=$2
HEMI=$3
NODE=${PBS_ARRAYID}
MODEL="visual"
#ALIGN="ha_testsubj"  #"ws"
ALIGN="ws"
STIM1="bg"
STIM2="actions"
STIM3="agents"


python ${MAINDIR}/PCATIKREG_03fullrange.py ${MODEL} ${ALIGN} ${STIM1} ${STIM2} ${STIM3} ${RUN} ${HEMI} ${SUB} ${NODE}
