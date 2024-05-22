this is a folder that keeps track of annotation and w2v files. Some input files are derived from Cara's repo, but we also generate our own GloVe embeddings. 

### annot_csv 
* Filetype: a csv that keeps a list of words per feature type.
* original location (1): `/idata/DBIC/cara/w2v/semantic_categories/`
* migrated to: `'/dartfs/rc/lab/D/DBIC/DBIC/life_data/cara/cara/w2v/semantic_categories'` -> now in our git repo here.

### filtered_stim: 
* Filetype: a csv file that keeps a list of words per TR. 
* Description: filtering has been applied: 1) non words 2) combined words such as venus_flytrap is split into two words.
* code: generated via `scripts/annot/extract_semantic_category_glove.py`

### glove
* Filetype: numpy array. Final output of extracting glove embeddings from semantic categories. 
* code: generated via `scripts/annot/extract_semantic_category_glove.py`

