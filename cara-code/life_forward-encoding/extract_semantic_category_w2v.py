import nltk
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors

import os, csv
cara_data_dir = '/idata/DBIC/cara/life'
tr_dict = {1:369, 2:341, 3:372, 4:406}

cat_dir = '/idata/DBIC/cara/w2v/semantic_categories/'
# Load pretrained model (since intermediate data is not included, the model cannot be refined with additional data)
google = KeyedVectors.load_word2vec_format('/idata/DBIC/cara/w2v/w2v_src/GoogleNews-vectors-negative300.bin', binary=True)

# c = np.load('/ihome/cara/life/w2v_src/google_w2v_ca.npy')

def create_w2v(stim):
    avg_google_vec = []
    flat_stim = [item for sublist in stim for item in sublist]
    for word in flat_stim:
        avg_google_vec.append(google[word])
    avg_goog = np.mean(np.column_stack(avg_google_vec), axis=1)[None, :]

    google_mean_vecs = []
    for i, tr in enumerate(stim):
        # if there are filtered words in this TR
        if len(tr) > 0:
            google_vec = []
            for word in tr:
                google_vec.append(google[word])
            goog_mean = np.mean(np.column_stack(google_vec), axis=1)[None, :]
        # no relevant words in this TR, make empty vector
        else:
            # goog_mean = np.zeros((1,300))
            goog_mean = avg_goog
        google_mean_vecs.append(goog_mean)
    goog = np.concatenate(google_mean_vecs, axis=0)
    print('google: {0}'.format(goog.shape))

    return(goog)

class TextLabeler():
    def __init__(self, text, lod):
        self.text = text
        self.iterate(lod)

    def replace_kv(self, _dict):
        """Replace any occurrence of a value with the key"""
        for key, value in _dict.iteritems():
            label = """{0}""".format(key)
            self.text = [[x.replace(value,label).strip() for x in l] for l in self.text]
        return self.text

    def iterate(self, lod):
        """Iterate over each dict object in a given list of dicts, `lod` """
        for _dict in lod:
            self.text = self.replace_kv(_dict)
        return self.text

lod = [{'deer':'dear'}, {'squawking':'squaking'}, {'highspeed':'high-speed'}, {'birdcall':'bird noise'}, {'':'noises'}, {'':'noise'}, \
    {'':'vocalizations'}, {'':'vocalization'}, {'':'sound'}, {'':'narration'}, \
    {'black screen':'blackscreen'}, {'crab eating':'crab-eater'}, {'orcas':'killer whales'}, \
    {'orca':'killer whale'}, {'seal':'elephant seal'}, {'poison_dart_frog':'poison arrow frog'}, \
    {'tree_trunk':'tree trunk'}, {'tree_trunks':'tree trunks'}, {'dorsal_fins':'dorsal fins'}, \
    {'poison_dart_frog':'poison arrow        frog'}, {'dorsal_fin':'dorsal fin'}, {'':','}, {'chinstrap_penguin':'chinstrap penguin'}, \
    {'zoom':'zoomed in'},{'zoom':'zoomed out'}, {'blue_sky':'blue_sky'}, {'forked_tongue':'forked tongue'}, \
    {'rattlesnake':'rattle snake'}, {'areca_palm':'nut palm plant'}, {'tidepool':'tidal pool'}, \
    {'praying_mantis':'praying mantis'}, {'venus_flytrap':'venus fly trap'}, {'leopard_seal':'leopard seal'}, \
    {'spiderweb':'spider web'}, {'':' but'}, {'':'music'}, {'caging':'encaging'}, \
    {'seafloor':'sea_floor'}, {'silhouette':'silhoutte'}, {'swinging':'brachiating'}, {'lifelong':'life-long'}, \
    {'fast swimming':'fast-swimming'}, {'learned':'learnt'}, {'Madagascar':'madagascar'}, {'Bandhavgarh':'bandhavgarh'}, \
    {'defense':'defence'}, {'neighborhood':'neighbourhood'}, {'neighbor':'neighbour'}, {'Falkland':'falkland'}, \
    {'meters':'metres'}, {'favorite':'favourite'}, {'color':'colour'}, {'Luangwa':'luangwa'}, {'fertilized':'fertilised'}, \
    {'center':'centre'}, {'Oregon':'oregon'}, {'December':'december'}, {'India':'india'}, {'Gentoo':'gentoo'}, \
    {'Antarctic':'antarctic'}, {'California':'california'}, {'Florida':'florida'}, {'Antarctica':'antarctica'}, \
    {'Zambia':'zambia'}, {'Brazil':'brazil'}, {'Kenya':'kenya'}, {'Malaysia':'malaysia'}, {'Pacific':'pacific'}]

# flat_ca = [[],[],[],[]]
# times = [[],[],[],[]]
# pos = [[],[],[],[]]
# directory = os.path.join(cara_data_dir, 'annotations')
# for f in os.listdir(directory):
#     if 'csv' in f:
#         print(f)
#         run = int(f[-5])
#         ds = pd.read_csv(os.path.join(directory, f), header=None)
#         ds = ds.dropna(axis=0, how='any')
#         word_list = ds[2].tolist()
#         print(len(word_list))
#         flat_ca[run-1] = word_list
#         time = np.array(ds[1].astype(float))
#         time = time / 1000.
#         times[run-1] = time
#         pos[run-1] = [str(line).split(' ')[2] for line in ds[0]]

vecs = []
times = []
durs = []
flat_ca = []
for i in range(1,5):
    with open('/idata/DBIC/cara/w2v/src/json_w2v/run{run}.json'.format(run=i), 'r') as f:
            contents = f.read()
    vec = contents.split('AvgVector')[1:]
    vec = [_.split('[')[1].split(']')[0] for _ in vec]
    vec = np.array([np.fromstring(_, sep=' ') for _ in vec])
    vecs.append(vec)

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

# verbs = []
# nouns = []
# for i in range(4):
#     for j,p in enumerate(pos[i]):
#         if 'VB' in p:
#             verbs.append(flat_ca[i][j].lower())
#         elif 'NN' in p:
#             nouns.append(flat_ca[i][j].lower())
#
# verbs = list(set(verbs))
# nouns = list(set(nouns))

# #Assuming res is a flat list
# with open('narrative_nouns.csv', 'w') as output:
#     writer = csv.writer(output, lineterminator='\n')
#     for val in nouns:
#         writer.writerow([val])

stim = []
for i in range(4):
    end = times[i][-1] + durs[i][-1]
    # end = times[i][-1]
    tr_list = np.arange(0, times[i][-1], 2.5)
    stim_run = [[0 for x in range(1)] for x in range(tr_list.shape[0])]
    prev_ind = 0
    ca_dict = {}
    for t in range(tr_list.shape[0]):
        if prev_ind not in ca_dict:
            ca_dict[prev_ind] = 0
        ca_dict[prev_ind] += 1
        # print(keys[-1])
        ind = np.searchsorted(times[i], tr_list[t])
        # print(tr[t], w2v[ind,:], ind)
        if ind == 0:
            stim_run[t] = flat_ca[i][0]
            # stim_run[t] = []
        else:
            # stim_run[t] = flat_ca[i][ind-1]
            stim_run[t] = [item for sublist in flat_ca[i][prev_ind:ind+1] for item in sublist]
        prev_ind = ind
    print('stim shape: {0} for run {1}'.format(len(stim_run), i))
    stim.append(stim_run)

print('Done loading runfiles')
# for run in range(1,5):
#     while len(stim[run-1]) < tr_dict[run]:
#         stim[run-1].append([])
#     print(len(stim[run-1]), tr_dict[run])

semantic_data = [item for sublist in stim for item in sublist]
durs = [item for sublist in durs for item in sublist]

filtered_words= []
for tr in semantic_data:
    word_list = []
    for word in tr:
        lower_word = word.lower()
        if lower_word not in stopwords.words('english'):
            if not lower_word[-1].isalpha():
                word_list.append(lower_word[:-1])
            elif lower_word[-2:] == '\'s':
                word_list.append(lower_word[:-2])
            else:
                word_list.append(lower_word)
    filtered_words.append(word_list)

semantic_data = filtered_words
processed = TextLabeler(semantic_data, lod)
proc = [[words for segments in i for words in segments.split() if len(words)>2] for i in processed.text]

goog = create_w2v(proc)
np.save('/idata/DBIC/cara/w2v/new_annotations/visual_all.npy', goog)

def make_list(file):
    with open(file, 'rb') as f:
        reader = csv.reader(f)
        nested = list(reader)
    return [word for f in nested for word in f]

def save_semantic_combo_vec(modality, input_list):
    filt = []
    filepath = modality
    for input in input_list:
        filt.extend(make_list(input))
        filepath = '_'.join([filepath, input.split('_')[-1][:-4]])
    filtered_stim = [[word for word in tr if word in filt] for tr in proc]
    print(len(set(x for l in filtered_stim for x in l)))
    goog = create_w2v(filtered_stim)
    np.save('/idata/DBIC/cara/w2v/new_annotations/{0}.npy'.format(filepath), goog)

print('Data through initial processing, see proc')
for cat in os.listdir(cat_dir):
    if 'csv' in cat:
        filt = make_list(os.path.join(cat_dir, cat))
        filtered_stim = [[word for word in tr if word in filt] for tr in proc]
        goog = create_w2v(filtered_stim)
        np.save('/idata/DBIC/cara/w2v/new_annotations/{0}.npy'.format(cat.split('.')[0]), goog)

# let's make the pairwise models
# for i in ['visual_actions.csv', 'agents.csv', 'bg.csv']:
#     for j in ['actions.csv', 'agents.csv', 'bg.csv']:

for modality in ['visual', 'narrative']:
    input_list = [os.path.join(cat_dir, '_'.join([modality, 'actions.csv'])), os.path.join(cat_dir, '_'.join([modality, 'agents.csv'])), os.path.join(cat_dir, '_'.join([modality, 'bg.csv']))]
    save_semantic_combo_vec(modality, input_list)
    for i, type1 in enumerate(['actions.csv', 'agents.csv', 'bg.csv']):
        for j, type2 in enumerate(['actions.csv', 'agents.csv', 'bg.csv']):
            if type1 is not type2 and i < j:
                input_list = [os.path.join(cat_dir, '_'.join([modality, type1])), os.path.join(cat_dir, '_'.join([modality, type2]))]
                save_semantic_combo_vec(modality, input_list)
