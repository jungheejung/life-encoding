#!/bin/bash -l
#PBS -N life_encode_submit_py
#PBS -q default
#PBS -l nodes=1:ppn=4
#PBS -l walltime=24:00:00
#PBS -m bea
#PBS -A DBIC

subjects=("ws" "aa" "ha_common" "ha_testsubj")

for ALIGN in ${subjects[*]}; do
for RUN in {0..3}; do
mksub -F "${ALIGN} ${RUN}" /dartfs/rc/lab/D/DBIC/DBIC/f0042x1/life-encoding/scripts/s03_ridge-regression/c02_submitpython.sh
echo "Submitted job for ${ALIGN} - run ${RUN}"
done
done
