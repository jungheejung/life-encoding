#!/bin/bash -l
#PBS -N fe1aja
#PBS -q default
#PBS -l nodes=1:ppn=16
#PBS -l walltime=8:00:00
#PBS -A DBIC
#PBS -t 13,16,18-24
#PBS -l mem=50gb,vmem=60gb

cd $PBS_O_WORKDIR

module load python/2.7-Anaconda
source /optnfs/common/miniconda3/etc/profile.d/conda.sh
conda activate haxby_mvpc

#MAINDIR=/dartfs/rc/lab/D/DBIC/DBIC/f0042x1/life-encoding/scripts/s03_banded_cara-alpha_loro

echo "PBSARRAY: " ${PBS_ARRAYID}
# INFILE=`awk "NR==${PBS_ARRAYID}" ${MAINDIR}/canonical_sublist.txt`
INFILE=`awk "NR==${PBS_ARRAYID}" ./node_test.txt`
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



MODEL="visual"
#ALIGN="ha_testsubj"  #"ws"
ALIGN="ws"
STIM1="bg"
STIM2="actions"
STIM3="agents"

# COMMENT OUT
#SUB="sub-rid000024"
#RUN=3
#HEMI="lh"
#NODE=1081


python ./PCATIKREG_04fullrange_10000.py ${MODEL} ${ALIGN} ${STIM1} ${STIM2} ${STIM3} ${RUN} ${HEMI} ${SUB} ${NODE}
