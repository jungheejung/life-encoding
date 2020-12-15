#!/bin/bash -l 
#PBS -N fe1aja
#PBS -q default
#PBS -l nodes=1:ppn=16
#PBS -l walltime=12:00:00
#PBS -A DBIC
#PBS -t 1-10

cd $PBS_O_WORKDIR

module load python/2.7-Anaconda
source /optnfs/common/miniconda3/etc/profile.d/conda.sh
conda activate haxby_mvpc

MAINDIR=/dartfs/rc/lab/D/DBIC/DBIC/f0042x1/life-encoding/scripts/s03_banded_cara-alpha_loro

echo ${PBS_ARRAYID}
echo "PBSARRAY"
# INFILE=`awk "NR==${PBS_ARRAYID}" ${MAINDIR}/canonical_sublist.txt`
INFILE=`awk "NR==${PBS_ARRAYID}" ${MAINDIR}/sublist.txt`
SUB=$(echo $INFILE | cut -f1 -d,)
RUN=$(echo $INFILE | cut -f2 -d,)
HEMI=$(echo $INFILE | cut -f3 -d,)
MODEL="visual"
#ALIGN="ha_testsubj"  #"ws"
ALIGN="aa"
STIM1="bg"
STIM2="actions"
STIM3="agents"

python ${MAINDIR}/j02_banded_ridge.py  ${MODEL} ${ALIGN} ${STIM1} ${STIM2} ${STIM3} ${RUN} ${HEMI} ${SUB}

