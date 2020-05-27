# the durations of the Life movie runs are 374, 346, 377, 412 TRs (not seconds- each TR is 2.5 s)
# so 1509 TRs total; data is 1509 TRs x 40962 surface nodes
# CVU Jan 2018

import numpy as np
import sys, os, json, pprint

# data_dir = '/idata/DBIC/cara/life'
# s_dir = '/idata/DBIC/snastase/life'

# json_dir = os.path.join(data_dir, 'src', 'json_w2v')
json_dir = '/Users/caravanuden/Desktop/life/json_w2v'

n_samples = 1509
n_vertices = 40962
n_proc = 32     # how many cores do we have?
n_medial = {'lh': 3486, 'rh': 3491}


# load json w2v embeddings for each part (4 files)
# using json bc has vectors labeled with words
# use np.load("*.pkl") for pickled files instead
actions = []
for i in range(1,5):

    with open(os.path.join(json_dir, 'run{run}.json'.format(run=i)), 'r') as f:
        contents = f.read()

    # vec = contents.split('AvgVector')[1:]
    # vec = [_.split('[')[1].split(']')[0] for _ in vec]
    # vec = np.array([np.fromstring(_, sep=' ') for _ in vec])
    # np.save(os.path.join(data_dir, 'w2v', 'run{run}-w2vectors.npy'.format(run=i)), vec)
    #
    # time = contents.split('Start_Time:')[1:]
    # time = [_.split('\n')[0] for _ in time]
    # time = [float(_.split(':')[0]) * 60 + float(_.split(':')[1:][0]) if ':' in _ else float(_) for _ in time]
    # time = np.array([_ for _ in time])
    # np.save(os.path.join(data_dir, 'w2v', 'run{run}-times.npy'.format(run=i)), time)
    #
    # dur = contents.split('Duration:')[1:]
    # dur = [_.split('\n')[0] for _ in dur]
    # dur = [float(_.split(':')[0]) * 60 + float(_.split(':')[1:][0]) if ':' in _ else float(_) for _ in dur]
    # dur = np.array([_ for _ in dur])
    # np.save(os.path.join(data_dir, 'w2v', 'run{run}-durations.npy'.format(run=i)), dur)

    words = contents.split('Words')[1:]
    words = [_.split('[')[1].split(']')[0] for _ in words]
    words = [_.split(',') for _ in words]
    words = [j for i in words for j in i]
    words = [_.strip('\' ') for _ in words]
    words = list(set(words))
    print(len(words))
    action = [_ for _ in words if _[-3:] == 'ing']
    # other = [_ for _ in words if _[-3:] != 'ing']
    actions.append(action)
    # print('\n\n{0}'.format(other))

# run 1
unknown1 = ['red', 'fast', 'left']

objects1 = ['iceberg', 'bark', 'spiderweb', 'leaf', 'bubble', \
            'seaweed',  'circle', 'silt', 'branch', 'food', \
            'tree', 'word', 'cloud', 'rock', 'steam', \
            'flower', 'stick', 'snow', 'plant', 'wave']

animal1 = ['antelope', 'seal', 'dragonfly', 'giraffe', 'wing', \
            'fish', 'dorsal_fin', 'zebra', 'seal', 'killer_whale', \
            'praying_mantis', 'bird', 'leg', 'ostrich', 'feet', \
            'whale', 'eye', 'monkey', 'frog', 'bug', \
            'lizard', 'dolphin', 'head', 'ant', 'cheetah', \
            'cricket', 'insect', 'chameleon', 'tongue', 'hummingbird', \
            'face', 'teeth']

place1 = ['underwater', 'coast', 'seabed', \
            'rainforest', 'plains', 'mountains', 'trail', \
            'reef', 'savanna', 'ocean', 'light', \
            'forest', 'arctic', 'Earth', 'ice', 'water','grass']

# run 2
unknown2 = []

objects2 = ['rock', 'snakeskin', 'stick', 'tree', 'tree_trunk', \
            'branch', 'wave', 'seaweed', 'water', 'shrubbery', \
            'leaf', 'stick']

animal2 = ['seal', 'bird', 'squirrel', 'tail', 'monkey', \
            'penguin', 'killer_whale', 'rattlesnake', 'dorsal_fin', \
            'tongue', 'feet', 'eye', 'leg', 'tiger', \
            'bug', 'penguin', 'deer', 'head', 'teeth', 'moth', \
            'tail', 'young']

place2 = ['hill', 'sand', 'prairie', 'dirt', 'sunlight', \
            'mound', 'grass', 'forest', 'beach', \
            'ocean', 'shore', 'wind']

# run 3
unknown3 = ['carcass', 'dead', 'open', 'closed', \
            'pitted', 'fly', 'gray',  'green']

object3 = ['bark', 'stone', 'leaf', 'tree', 'nut', \
            'plant', 'shadow', 'cloud', 'rock', 'peelings', \
            'tree_trunk', 'water', 'shrubbery', 'droplets', 'branch', ]

animal3 = ['eye', 'bird', 'tusk', 'swordfish', 'hand', \
            'monkey', 'fish', 'hair', 'bird', 'leg', \
            'insect', 'venus_flytrap', 'head', 'lion', 'hippo', \
            'lion_cub', 'capuchin', 'mouth']

place3 = ['underwater', 'river', 'sunrise', \
            'sun', 'mud', 'landscape', 'dirt', 'blue_sky', 'muddy', \
            'plains', 'flat', 'dry', 'mountains', 'field', \
            'forest', 'cliff', 'land', 'lakebed', 'ocean', 'shore', \
            'lagoon', 'barren']
# run 4
unknown4 = ['drop', 'back', 'dead']

object4 = ['tree_trunk', 'rock', 'leaf', 'tree', \
            'plant', 'wave', 'branch', 'stick', 'seaweed', \
            'shadow', 'puddle']

animal4 = ['bird', 'tail', 'tentacle', 'foot', 'tadpole', \
            'grebe', 'leg', 'head', 'egg', 'hand', \
            'orangutan', 'octopus', 'feet', 'eye', 'frog', \
            'leopard_seal', 'crab', 'fish', 'baby', 'penguin']

place4 = ['underwater', 'sky', 'lake', 'cloud', 'river', \
            'sun', 'ice', 'water', 'seafloor', 'rainforest', 'mist', 'mountains' \
            'iceberg', 'sticks', 'reeds', 'ferns', 'glacier', \
            'nuts', 'forest', 'branch', 'moss', 'cliff', \
            'land', 'ocean', 'shore','fog', 'den']


action1 = actions[0] + ['spray', 'splash']
action2 = actions[1] + ['startled']
action3 = actions[2] + ['smashed', 'cracked']
action4 = actions[3] + ['dance', 'splash', 'ripped', 'leaves', 'eaten']
print(actions[0])
w = actions[0] + unknown1 + animal1 + place1
print(set([i for i in w if w.count(i)>1]))

#     np.save(os.path.join(data_dir, 'w2v', 'run{run}-words.npy'.format(run=i)), words)
#
#     print('in part {0}, vector has dimensions {1} and duration has dimensions {2}'.format(i, vec.shape, dur.shape))
#
# # load fMRI data from both hemispheres of each subject (36 files)
# # reorganize and split into 4 runs with supplied TR delimits
# participants = ['sub-rid000037', 'sub-rid000001', 'sub-rid000033', 'sub-rid000024',
#                 'sub-rid000019', 'sub-rid000041', 'sub-rid000032', 'sub-rid000006',
#                 'sub-rid000009', 'sub-rid000017', 'sub-rid000005', 'sub-rid000038',
#                 'sub-rid000031', 'sub-rid000012', 'sub-rid000027', 'sub-rid000014',
#                 'sub-rid000034', 'sub-rid000036']
#
# hemispheres = ['lh', 'rh']
#
# # Load in surface data sets
# def load_datasets(participants, hemispheres, runs = [374, 346, 377, 412]):
#     for participant in participants:
#         for hemi in hemispheres:
#             ds = mv.gifti_dataset(os.path.join(s_dir,'{0}_task-life.{1}.tproject.gii'.format(
#                                participant, hemi)))
#             ds.sa.pop('intents')
#             ds.sa['subjects'] = [participant] * ds.shape[0]
#             ds.fa['node_indices'] = range(n_vertices)
#             # z-score features across samples
#             mv.zscore(ds, chunks_attr=None)
#             print('total ds: {0}'.format(ds.shape))
#             r1 = ds[0:374,:]
#             r2 = ds[374:720,:]
#             r3 = ds[720:1097,:]
#             r4 = ds[1097:1509,:]
#             print('r1: {0}'.format(r1.shape))
#             print('r2: {0}'.format(r2.shape))
#             print('r3: {0}'.format(r3.shape))
#             print('r4: {0}'.format(r4.shape))
#             mv.map2gifti(r1, filename=os.path.join(data_dir, 'fmri','{0}_task-life_run-1_hemi-{1}.tproject.gii'.format(
#                                 participant, hemi)))
#             mv.map2gifti(r2, filename=os.path.join(data_dir, 'fmri','{0}_task-life_run-2_hemi-{1}.tproject.gii'.format(
#                                 participant, hemi)))
#             mv.map2gifti(r3, filename=os.path.join(data_dir, 'fmri','{0}_task-life_run-3_hemi-{1}.tproject.gii'.format(
#                                 participant, hemi)))
#             mv.map2gifti(r4, filename=os.path.join(data_dir, 'fmri','{0}_task-life_run-4_hemi-{1}.tproject.gii'.format(
#                                 participant, hemi)))
#             print('finished loading/organizing files for participant {0}, hemi {1}'.format(participant,hemi))
#
# load_datasets(participants, hemispheres)
#
# # add in 1, 2, 3, 4 TR delay to stimulus data
# for i in range(1,5):
#     vec = np.load(os.path.join(data_dir, 'w2v', 'run{0}-w2vectors.npy'.format(i)))
#     time = np.load(os.path.join(data_dir, 'w2v', 'run{0}-times.npy'.format(i)))
#     dur = np.load(os.path.join(data_dir, 'w2v', 'run{0}-durations.npy'.format(i)))
#
#     tr = np.arange(0, time[-1] r+ dur[-1], 2.5)
#     # print('tr shape: {0}'.format(tr.shape))
#
#     stim = np.zeros((tr.shape[0], 300))
#     for t in range(tr.shape[0]):
#         # print(tr[t])
#         ind = np.searchsorted(time, tr[t])
#         # print(tr[t], w2v[ind,:], ind)
#
#         stim[t] = vec[ind-1]
#     print('stim shape: {0}'.format(stim.shape))
#
#     print('stim cutoff shape: {0}'.format(stim[3:-1].shape))
#     print('stim cutoff shape: {0}'.format(stim[2:-2].shape))
#     print('stim cutoff shape: {0}'.format(stim[1:-3].shape))
#     print('stim cutoff shape: {0}'.format(stim[:-4].shape))
#
#
#     stim = np.concatenate((stim[3:-1], stim[2:-2], stim[1:-3], stim[:-4]), axis=1)
#     print(stim.shape)
#
#     np.save(os.path.join(data_dir, 'w2v', 'run{run}-tr-w2vectors.npy'.format(run=i)), stim)

stim = []
for i in range(1,5):
    print(i)
    vec = json.load(open('/Users/caravanuden/Desktop/life/forward_encoding/old_codes/Part{0}_Raw_Data.json'.format(i)))
    if i == 1:
        pprint.pprint(vec['features'].keys())
    keys = np.array(list(vec['features'].keys()))
    vals = list(vec['features'].values())
    print(int(keys[-1])+2)
    tr_list = np.arange(0, int(keys[-1])+2, 2.5)

    print('tr shape: {0}'.format(tr_list.shape))
    stim_run = [[0 for x in range(1)] for x in range(tr_list.shape[0])]
    for t in range(tr_list.shape[0]):
        # print(tr[t])
        ind = np.searchsorted(keys, tr_list[t])
        # print(tr[t], w2v[ind,:], ind)

        stim_run[t] = vals[ind-1]

    print('stim shape: {0} for run {1}'.format(len(stim_run), i))
    stim.append(stim_run)

for run in stim:
    for tr in run:
        for word in tr:
            glove
