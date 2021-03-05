#!/bin/bash -l
#PBS -N fe1aja
#PBS -q default
#PBS -l nodes=1:ppn=16
#PBS -l walltime=24:00:00
#PBS -A DBIC
#PBS -t 1-4

cd $PBS_O_WORKDIR

module load python/2.7-Anaconda
source /optnfs/common/miniconda3/etc/profile.d/conda.sh
conda activate haxby_mvpc

#MAINDIR=/dartfs/rc/lab/D/DBIC/DBIC/f0042x1/life-encoding/scripts/s03_banded_cara-alpha_loro

echo "PBSARRAY: " ${PBS_ARRAYID}
# INFILE=`awk "NR==${PBS_ARRAYID}" ${MAINDIR}/canonical_sublist.txt`
INFILE=`awk "NR==${PBS_ARRAYID}" ./sublist_with_range.txt`
SUB=$(echo $INFILE | cut -f1 -d,)
RUN=$(echo $INFILE | cut -f2 -d,)
HEMI=$(echo $INFILE | cut -f3 -d,)
NODE=$(echo $INFILE | cut -f4 -d,)
# SUB=$1
# RUN=$2
# HEMI=$3
# NODE=${PBS_ARRAYID}
MODEL="visual"
#ALIGN="ha_testsubj"  #"ws"
ALIGN="ws"
STIM1="bg"
STIM2="actions"
STIM3="agents"


python ./PCATIKREG_03fullrange.py ${MODEL} ${ALIGN} ${STIM1} ${STIM2} ${STIM3} ${RUN} ${HEMI} ${SUB} ${NODE}
