#%% 
import nltk
nltk.download('stopwords')
import os, csv
import numpy as np
import pandas as pd
import gensim.downloader
from nltk.corpus import stopwords
from pathlib import Path
from os.path import join
import json
from nltk.corpus import stopwords
import matplotlib.pyplot as plt 
"""
From the annotations.json, we tranform this into glove embeddings

TODO: refactor for spacetop files (12 vs 4. different TRs)
change paths
* Annotation path: /dartfs/rc/lab/H/HaxbyLab/heejung/annotation_spacetopLLM
* .gii data: /dartfs/rc/lab/H/HaxbyLab/heejung/data_spacetoptrim

TODO 02/09/2026:
* empty list: vector of 0
* more than 0 items: get embeddings -> average embeddings
* deal with "person in spacesuit" -> split by space nested list. check if word is stop word. average items. 

Check wanderer: 
['astronaut', 'person in spacesuit'],


# * empty list: vector of 0 create_w2v
# * more than 0 items: get embeddings -> average embeddings
# * deal with "person in spacesuit" -> split by space nested list. check if word is stop 
# word. 
# * Filter stop words (space in black) -> space black

"""


file_dict = {
'ses-01_run-01_order-02_content-wanderers':430,
'ses-01_run-02_order-02_content-HB':123,
'ses-01_run-03_order-01_content-huggingpets':230,
'ses-01_run-03_order-04_content-dancewithdeath':295,
'ses-01_run-04_order-02_content-angrygrandpa':736,
'ses-02_run-02_order-03_content-menrunning':441,
'ses-02_run-03_order-01_content-unefille':252,
'ses-02_run-03_order-04_content-war':134,
'ses-03_run-02_order-01_content-planetearth':321,
'ses-03_run-02_order-03_content-heartstop':426,
'ses-03_run-03_order-01_content-normativeprosocial2':228,
'ses-04_run-01_order-02_content-gockskumara':389
}
# DUMMY = '/Users/h/Downloads/life/ses-01_run-01_order-02_content-wanderers_frames/analysis_results.json'

OUTPUT = '/Users/h/Downloads/life'
glove = gensim.downloader.load('glove-wiki-gigaword-300')
# %%
def filter_nonexistent_words(word):
            # glove_vec.append(glove[word])
    try:
        emb = glove[word]
        print(emb.shape)
    except KeyError:
        # emb = np.zeros((300))
        emb = np.full((300), np.nan)
    return emb

def create_glove(stim, remove_stopwords):
        stop_words = set(stopwords.words('english')) 
        if remove_stopwords:
            stop_words.update(remove_stopwords)

        glove_mean_vecs = []
        for i, tr in enumerate(stim):
            print(tr)
            # if there are filtered words in this TR
            if len(tr) > 0:

                glove_vec = []
                for word in tr:
                    # Case 1: Handle compound labels
                    if ' ' in word or '/' in word:
                        preproc_word = word.replace('/', ' ')
  
                        parts = [w for w in preproc_word.split() if w in glove]

                        if len(parts) > 0:  # Only process if we have valid words
                            vectors = [filter_nonexistent_words(w) for w in parts if w not in stop_words ] # Each is (300,)
                            # 1-1. vector is valid after filtering stopwords. Average them to create a single compound
                            if len(vectors) > 0:
                                compound_word = np.mean(vectors, axis=0) #(300,)
                                glove_vec.append(compound_word[np.newaxis, :])
                            # 1-2. vector is empty because all words were stopwords
                            else: 
                                glove_vec.append(np.full((1, 300), np.nan))
                        # 1-3. No tokens in the compound label exist in the GloVe vocabulary
                        else:
                            glove_vec.append(np.full((1, 300), np.nan))

                    # Case 2: single word
                    else:
                        if word not in stop_words:  # Check stopwords first
                            # 2-1. Get glove
                            try:
                                glove_vec.append(glove[word][np.newaxis,:])

                            except KeyError:
                                # 2-2. Word missing from GloVe
                                # If it's the only word, return zeros; ['zzz']
                                # otherwise use NaN to flag for averaging ['astronaut', 'zzz'] -> we dont want the 0 vector from zzz to cancel out astronaut

                                if len(tr) == 1:
                                    glove_vec.append(np.zeros((1, 300)))
                                else:
                                    glove_vec.append(np.full((1,300), np.nan))
                        else: # if the only word is a stopword
                            # Word is a stopword, skip or add NaN
                            glove_vec.append(np.full((1, 300), np.nan))
                # Compute mean if we have vectors
                if len(glove_vec) > 0:
                    goog_mean = np.mean(glove_vec, axis=0)
                else:
                    goog_mean = np.zeros((1, 300))
                # goog_mean = np.mean(glove_vec, axis=0)
                print(f"goog_mean: {goog_mean.shape}")
            # no relevant words in this TR, make empty vector
            else:
                goog_mean = np.zeros((1, 300))
            glove_mean_vecs.append(goog_mean)
        goog = np.stack(glove_mean_vecs, axis=0) #np.concatenate(glove_mean_vecs, axis=0)
        print('glove: {0}'.format(goog.shape))
        return(goog)


for FNAME, TR in file_dict.items():

    for FEATURE in ['agents', 'actions', 'scenes','objects']:
        with open(f"{OUTPUT}/{FNAME}_frames/analysis_results.json", 'r') as f:
                contents = f.read()

        data = json.loads(contents)

        timestamps_list = []
        for item in data:
            timestamps_list.append(item['timestamp'])
        # minimal difference
        tr_list = np.arange(0, 0.46*TR, 0.46)
        feature_list = []; 
        # 1: loop through nparange

        for current_tr in tr_list:
            #  2: find nearest number in json (timestamps)
            nearest_ts = min(timestamps_list, key=lambda x: abs(x - current_tr))
            json_index = timestamps_list.index(nearest_ts)
            # 3: compile word list
            # feature_list.append(data[json_index][FEATURE])
            # Error when annotation doesn't have scenes
            feature_list.append(data[json_index].get(FEATURE, []))


        remove_stopwords = ['in', 'Esther the Wonder Pig']
        
        # 4. Now run your original function
        # It will now find both 'astronaut' and 'space explorer' in your new dictionary
        final_output = create_glove(feature_list, remove_stopwords)
        new_arr = np.squeeze(final_output, axis=1)
        # plt.imshow(new_arr)
        print(f"{FNAME}: {TR} and {final_output.shape}")
        # 
        np.save(f'{OUTPUT}/{FNAME}_feature-{FEATURE}.npy', new_arr)


# 3. Create and save the plot
        plt.figure(figsize=(10, 8)) # Initialize a new figure
        plt.imshow(new_arr, aspect='auto', interpolation='nearest')
        plt.colorbar(label='GloVe Weight')
        plt.title(f"Heatmap: {FNAME} - {FEATURE}")
        plt.xlabel("Embedding Dimension (300)")
        plt.ylabel("Time (TR)")
        
        # Save as PNG
        plt.savefig(f'{OUTPUT}/{FNAME}_feature-{FEATURE}.png', dpi=300, bbox_inches='tight')
        
        # 4. Crucial: Close the plot to free up memory
        plt.close()

# %%

# %% SANDBOX
# import numpy as np
# annotation = [ [], 
#               ['astronaut', 'person'], 
#               ['astronaut', 'person in spacesuit' ],
#               ['astronaut']
#               ]

# person = create_glove([['person']], remove_stopwords)
# spacesuit = create_glove([['spacesuit']], remove_stopwords)

# compound = create_glove([['person in spacesuit']], remove_stopwords)
# assert np.array_equal(np.mean(np.vstack([person,spacesuit]), axis=0) , compound.flatten())



