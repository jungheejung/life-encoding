#PBS -N life_encode_cara_alpha
#PBS -q default
#PBS -l nodes=1:ppn=8
#PBS -l walltime=05:00:00
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
MAINDIR=/dartfs/rc/lab/D/DBIC/DBIC/f0042x1/life-encoding/scripts/s03_banded_cara-alpha_loro
SUB=${1}
RUN=${2}
HEMI=${3}

python ${MAINDIR}/c03_banded_ridge.py  ${MODEL} ${ALIGN} ${STIM1} ${STIM2} ${RUN} ${HEMI} ${SUB}
echo "Running $CMD"
