for sub in sub-*/; do
    count=0
    for ses in ses-01 ses-02 ses-03 ses-04; do 
    [ -d "${sub}/${ses}" ] && ((count++))
    done 

    if [ "$count" -ne 4 ]; then 
    echo "$sub does not have all 4 sessions"
    fi
done