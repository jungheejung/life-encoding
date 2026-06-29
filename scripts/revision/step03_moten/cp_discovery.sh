for f in /Users/h/Documents/projects_local/life-encoding/data/revision_moten/moten_video-*.npy; do
    fname=$(basename "$f")
    new_fname="${fname/moten_video-/}_feature-moten.npy"
    new_fname="${new_fname/.npy_feature-moten.npy/_feature-moten.npy}"
    cp "$f" "ssh f0042x1@discovery8.dartouth.edu:/dartfs/rc/lab/H/HaxbyLab/heejung/data/annotations/glove/${new_fname}"
done

# scp originals over first
scp /Users/h/Documents/projects_local/life-encoding/data/revision_moten/moten_video-*.npy \
    f0042x1@discovery8.dartouth.edu:/dartfs/rc/lab/H/HaxbyLab/heejung/data/annotations/glove/

# then ssh in and rename
ssh f0042x1@discovery8.dartouth.edu 'for f in /dartfs/rc/lab/H/HaxbyLab/heejung/data/annotations/glove/moten_video-*.npy; do
    new_fname=$(basename "$f" | sed "s/^moten_video-//; s/\.npy$/_feature-moten.npy/")
    mv "$f" "/dartfs/rc/lab/H/HaxbyLab/heejung/data/annotations/glove/$new_fname"
done'