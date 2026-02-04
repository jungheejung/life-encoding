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
"""
From the annotations.json, we tranform this into glove embeddings

TODO: refactor for spacetop files (12 vs 4. different TRs)
change paths
* Annotation path: /dartfs/rc/lab/H/HaxbyLab/heejung/annotation_spacetopLLM
* .gii data: /dartfs/rc/lab/H/HaxbyLab/heejung/data_spacetoptrim


"""
DUMMY = '/Users/h/Downloads/life/ses-01_run-01_order-02_content-wanderers_frames/analysis_results.json'
END = 430
with open(DUMMY, 'r') as f:
        contents = f.read()

data = json.loads(contents)
# %%
timestamps_list = []
for item in data:
#     print(item)
    timestamps_list.append(item['timestamp'])
#     # feature_list.append(item['agents'])
# %% minimal difference
tr_list = np.arange(0, 0.46*END, 0.46)
feature_list = []; 
# 1: loop through nparange

for current_tr in tr_list:
    #  2: find nearest number in json (timestamps)
    nearest_ts = min(timestamps_list, key=lambda x: abs(x - current_tr))
    json_index = timestamps_list.index(nearest_ts)
    # 3: compile word list
    feature_list.append(data[json_index]['agents'])


glove = gensim.downloader.load('glove-wiki-gigaword-300')

def create_w2v(stim):

    glove_mean_vecs = []
    for i, tr in enumerate(stim):
        # if there are filtered words in this TR
        if len(tr) > 0:
            glove_vec = []
            for word in tr:
                # glove_vec.append(glove[word])
                glove_vec.append(glove[word])
            goog_mean = np.mean(np.column_stack(glove_vec), axis=1)[None, :]
        # no relevant words in this TR, make empty vector
        else:
            goog_mean = np.zeros((1,300))
            # goog_mean = avg_goog
        glove_mean_vecs.append(goog_mean)
    goog = np.concatenate(glove_mean_vecs, axis=0)
    print('glove: {0}'.format(goog.shape))

    return(goog)













#########################################################
def create_w2v(stim):

    glove_mean_vecs = []
    for i, tr in enumerate(stim):
        # if there are filtered words in this TR
        if len(tr) > 0:
            glove_vec = []
            for word in tr:
                # glove_vec.append(glove[word])
                glove_vec.append(glove[word])
            goog_mean = np.mean(np.column_stack(glove_vec), axis=1)[None, :]
        # no relevant words in this TR, make empty vector
        else:
            goog_mean = np.zeros((1,300))
            # goog_mean = avg_goog
        glove_mean_vecs.append(goog_mean)
    goog = np.concatenate(glove_mean_vecs, axis=0)
    print('glove: {0}'.format(goog.shape))

    return(goog)

class TextLabeler():
    def __init__(self, text, lod):
        self.text = text
        self.iterate(lod)

    def replace_kv(self, _dict):
        """Replace any occurrence of a value with the key"""
        # for key, value in _dict.iteritems():
        for key, value in _dict.items():
            label = """{0}""".format(key)
            self.text = [[x.replace(value,label).strip() for x in l] for l in self.text]
        return self.text

    def iterate(self, lod):
        """Iterate over each dict object in a given list of dicts, `lod` """
        for _dict in lod:
            self.text = self.replace_kv(_dict)
        return self.text

def make_list(file):
    with open(file, 'rb') as f:
        # reader = csv.reader(f)
        reader = pd.read_csv(f, header=None)
        nested = list(reader[0])
    return nested #[word for f in nested for word in f]



times = []
durs = []
flat_ca = []
for i in range(1,5):
    # with open('/idata/DBIC/cara/w2v/src/json_w2v/run{run}.json'.format(run=i), 'r') as f:
    with open(f'/dartfs/rc/lab/D/DBIC/DBIC/life_data/cara/cara/w2v/src/json_w2v/run{i}.json', 'r') as f:
            contents = f.read()

    time = contents.split('Start_Time:')[1:]
    time = [_.split('\n')[0] for _ in time]
    time = [float(_.split(':')[0]) * 60 + float(_.split(':')[1:][0]) if ':' in _ else float(_) for _ in time]
    time = np.array([_ for _ in time])
    time = np.subtract(time, time[0])
    times.append(time)

    dur = contents.split('Duration:')[1:]
    dur = [_.split('\n')[0] for _ in dur]
    dur = [float(_.split(':')[0]) * 60 + float(_.split(':')[1:][0]) if ':' in _ else float(_) for _ in dur]
    dur = np.array([_ for _ in dur])
    durs.append(dur)

    words = contents.split('Words')[1:]
    words = [_.split('[')[1].split(']')[0] for _ in words]
    words = [_.split(',') for _ in words]
    words =[[_.strip(' ') for _ in sl] for sl in words]
    words =[[_.strip('\'') for _ in sl] for sl in words]

    print(i)
    print(['{0}\n'.format(word) for word in words])
    flat_ca.append(words)


stim = []
for i in range(4):
    end = times[i][-1] + durs[i][-1]
    print(f"end for each run {i}: {end}")
    print(f"end for each run {i}: {end/2.5}")
    tr_list = np.arange(0, end, 2.5)
    stim_run = [[] for x in range(tr_list.shape[0])]
    for t in range(tr_list.shape[0]):
        ind = np.searchsorted(times[i], tr_list[t])
        if ind == 0:
            stim_run[t] = flat_ca[i][0]
        else:

            stim_run[t] = flat_ca[i][ind-1]#[item for sublist in flat_ca[i][ind] for item 
    print('stim shape: {0} for run {1}'.format(len(stim_run), i))
    stim.append(stim_run)

print('Done loading runfiles')

semantic_data = [item for sublist in stim for item in sublist]

#### check which words are not part of Glove
unique_values = set(item for sublist in semantic_data for item in sublist)
error_list_test = []
for word in unique_values:
    try:
        glove[word]
    except KeyError as e:
        error_list_test.append(word)
####
filtered_words= []
stopwords_list = stopwords.words('english')
for tr in semantic_data:
    word_list = []
    for word in tr:
        lower_word = word.lower()
        if lower_word not in stopwords_list:
        # if lower_word not in stopwords.words('english'):
            if not lower_word[-1].isalpha():
                word_list.append(lower_word[:-1])
            elif lower_word[-2:] == '\'s':
                word_list.append(lower_word[:-2])
            elif '_' in lower_word:
                word_list.extend(lower_word.split('_'))
            else:
                word_list.append(lower_word)
    filtered_words.append(word_list)

##### get unique words from processed.text
filtered_unique = set(item for sublist in filtered_words for item in sublist)

error_list = []
for word in filtered_unique:
    try:
        glove[word]
    except KeyError as e:
        error_list.append(word)


# goog = create_w2v(proc)
goog = create_w2v(filtered_words)

np.save('/dartfs/rc/lab/D/DBIC/DBIC/f0042x1/life-encoding/data/annotations/glove/visual_all.npy', goog)


print('Data through initial processing, see proc')
stim_savedir = '/dartfs/rc/lab/D/DBIC/DBIC/f0042x1/life-encoding/data/annotations/filtered_stim'
Path(stim_savedir).mkdir(parents=True, exist_ok=True)
for cat in os.listdir(cat_dir):
    if 'csv' in cat:
        filt = make_list(os.path.join(cat_dir, cat))
        # filtered_stim = [[word for word in tr if word in filt] for tr in proc]
        filtered_stim = [[word for word in tr if word in filt] for tr in filtered_words]

        pd.DataFrame(filtered_stim).to_csv(join(stim_savedir, cat))
        glove_features = create_w2v(filtered_stim)
        # np.save('/idata/DBIC/cara/w2v/new_annotations/{0}.npy'.format(cat.split('.')[0]), goog)
        np.save('/dartfs/rc/lab/D/DBIC/DBIC/f0042x1/life-encoding/data/annotations/glove/{0}.npy'.format(cat.split('.')[0]), glove_features)
