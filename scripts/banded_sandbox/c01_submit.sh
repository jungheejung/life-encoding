#!/bin/bash -l
#PBS -N life_encode_submit_py
#PBS -q default
#PBS -l nodes=1:ppn=4
#PBS -l walltime=24:00:00
#PBS -m bea
#PBS -A DBIC

subjects=("sub-rid000001")
# "sub-rid000005" "sub-rid000006" "sub-rid000009" "sub-rid000012" \
#"sub-rid000014" "sub-rid000017" "sub-rid000019" "sub-rid000024" "sub-rid000027" \
#"sub-rid000031" "sub-rid000032" "sub-rid000033" "sub-rid000034" "sub-rid000036" \
#"sub-rid000037" "sub-rid000038" "sub-rid000041")
# sub-rid000001 sub-rid000005

for SUB in ${subjects[*]}; do
mksub /dartfs/rc/lab/D/DBIC/DBIC/f0042x1/life-encoding/scripts/s03_banded-ridge/c02_submitpython.sh -F ${SUB} 
echo "Submitted job for ${SUB}"
done

