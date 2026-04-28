    for ofile in *.o; do
        base="${ofile%.o}"
        if ! grep -q "Finished running banded ridge regression!" "$ofile"; then
        echo "=== Last 3 lines of ${base}"
        head -n 1 "${base}.o"
        tail -3 "${base}.e"
        fi

        if grep -q "CANCELLED AT" "${base}.e"; then
        JOB=${base##*_}
        echo $JOB >> /vast/labs/DBIC/datasets/Life/life-encoding/scripts/revision/step04_himalaya/RERUN_SCENES_JOBS.txt
        fi
    done


# TODO: 
# when resubmitting, purge previous log files with same job id