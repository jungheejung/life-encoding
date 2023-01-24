#!/bin/bash -l
#PBS -N life_encode_submit_py
#PBS -q default
#PBS -l nodes=1:ppn=4
#PBS -l walltime=24:00:00
#PBS -m bea
#PBS -A DBIC

subjects=("sub-rid000005" "sub-rid000006")

for SUB in ${subjects[*]}; do
for RUN in {0,2,3}; do
mksub -F "${SUB} ${RUN}" /dartfs/rc/lab/D/DBIC/DBIC/f0042x1/life-encoding/scripts/s03_banded-ridge/c02_submitpython.sh
echo "Submitted job for ${SUB} - run ${RUN}"
done
done

# mksub -F "sub-rid000009 3" /dartfs/rc/lab/D/DBIC/DBIC/f0042x1/life-encoding/scripts/s03_banded-ridge/c02_submitpython.sh
# echo "Submitted job for sub-rid000009 - run 3"
# mksub -F "sub-rid000012 0" /dartfs/rc/lab/D/DBIC/DBIC/f0042x1/life-encoding/scripts/s03_banded-ridge/c02_submitpython.sh
# echo "Submitted job for sub-rid000012 - run 0"
# mksub -F "sub-rid000012 1" /dartfs/rc/lab/D/DBIC/DBIC/f0042x1/life-encoding/scripts/s03_banded-ridge/c02_submitpython.sh
# echo "Submitted job for sub-rid000012 - run 1"

