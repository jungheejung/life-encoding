from gensim.models import KeyedVectors
from os.path import join
import numpy as np

# # Load pretrained model (since intermediate data is not included, the model cannot be refined with additional data)
google = KeyedVectors.load_word2vec_format('/ihome/cara/life/forward-encoding/GoogleNews-vectors-negative300.bin', binary=True)

"""
word2vec embeddings start with a line with the number of lines (tokens?) and
the number of dimensions of the file. This allows gensim to allocate memory
accordingly for querying the model. Larger dimensions mean larger memory is
held captive. Accordingly, this line has to be inserted into the GloVe
embeddings file.
"""

import os
import shutil
import hashlib
from sys import platform
import gensim, json

def prepend_line(infile, outfile, line):
				"""
				Function use to prepend lines using bash utilities in Linux.
				(source: http://stackoverflow.com/a/10850588/610569)
				"""
				with open(infile, 'r') as old:
						with open(outfile, 'w') as new:
								new.write(str(line) + "\n")
								shutil.copyfileobj(old, new)

def prepend_slow(infile, outfile, line):
				"""
				Slower way to prepend the line by re-creating the inputfile.
				"""
				with open(infile, 'r') as fin:
						with open(outfile, 'w') as fout:
								fout.write(line + "\n")
								for line in fin:
												fout.write(line)

def checksum(filename):
				"""
				This is to verify the file checksum is the same as the glove files we use to
				pre-computed the no. of lines in the glove file(s).
				"""
				BLOCKSIZE = 65536
				hasher = hashlib.md5()
				with open(filename, 'rb') as afile:
						buf = afile.read(BLOCKSIZE)
						while len(buf) > 0:
								hasher.update(buf)
								buf = afile.read(BLOCKSIZE)
				return hasher.hexdigest()

# Input: GloVe Model File
# More models can be downloaded from http://nlp.stanford.edu/projects/glove/
glove_file="glove.840B.300d.txt"
print(glove_file.split('.'))
_, tokens, dimensions, _ = glove_file.split('.')
num_lines = 2196017
dims = int(dimensions[:-1])

# Output: Gensim Model text format.
gensim_file='glove_model.txt'
gensim_first_line = "{} {}".format(num_lines, dims)

# Prepends the line.
if platform == "linux" or platform == "linux2":
				prepend_line(glove_file, gensim_file, gensim_first_line)
else:
				prepend_slow(glove_file, gensim_file, gensim_first_line)

# Demo: Loads the newly created glove_model.txt into gensim API.
glove=gensim.models.KeyedVectors.load_word2vec_format(gensim_file,binary=False) #GloVe Model


dog = glove['dog']
print(dog.shape)
print(dog[:10])

# d_dir = '/idata/DBIC/cara/life'
# s_dir = '/idata/DBIC/snastase/life'
#
# json_dir = join(d_dir, 'src', 'json_w2v')

# load json w2v embeddings for each part (4 files)
# using json bc has vectors labeled with words
# use np.load("*.pkl") for pickled files instead
a = set()
o = set()
vecs = []
times = []
durs = []
flat_ca = []
for i in range(1,5):
	with open('/idata/DBIC/cara/life/src/json_w2v/run{run}.json'.format(run=i), 'r') as f:
			contents = f.read()
	vec = contents.split('AvgVector')[1:]
	vec = [_.split('[')[1].split(']')[0] for _ in vec]
	vec = np.array([np.fromstring(_, sep=' ') for _ in vec])
	vecs.append(vec)

# print(vec)
	# np.save(os.path.join(data_dir, 'w2v', 'run{run}-w2vectors.npy'.format(run=i)), vec)

	time = contents.split('Start_Time:')[1:]
	time = [_.split('\n')[0] for _ in time]
	time = [float(_.split(':')[0]) * 60 + float(_.split(':')[1:][0]) if ':' in _ else float(_) for _ in time]
	time = np.array([_ for _ in time])
	print(time[0])
	time = np.subtract(time, time[0])
	print(time[0])
	times.append(time)

	# np.save(os.path.join(data_dir, 'w2v', 'run{run}-times.npy'.format(run=i)), time)

	dur = contents.split('Duration:')[1:]
	dur = [_.split('\n')[0] for _ in dur]
	dur = [float(_.split(':')[0]) * 60 + float(_.split(':')[1:][0]) if ':' in _ else float(_) for _ in dur]
	dur = np.array([_ for _ in dur])
	durs.append(dur)

# print(dur)
# 	np.save(os.path.join(data_dir, 'w2v', 'run{run}-durations.npy'.format(run=i)), dur)

	words = contents.split('Words')[1:]
	words = [_.split('[')[1].split(']')[0] for _ in words]
	words = [_.split(',') for _ in words]
	words =[[_.strip(' ') for _ in sl] for sl in words]
	words =[[_.strip('\'') for _ in sl] for sl in words]

	flat_ca.append(words)
				# words = [j for i in words for j in i]
				# words = [_.strip('\' ') for _ in words]
				# run_words.extend(words)
				# words = list(set(words))
				# print(len(words))
				# action = [_ for _ in words if _[-3:] == 'ing']
				# other = [_ for _ in words if _[-3:] != 'ing']
				# a.update(action)
				# o.update(other)
				# print('\n\n{0}'.format(other))

stim = []
for i in range(4):
	tr_list = np.arange(0, times[i][-1] + durs[i][-1], 2.5)
	print(times[i][-1] + durs[i][-1], len(tr_list))
	stim_run = [[0 for x in range(1)] for x in range(tr_list.shape[0])]
	for t in range(tr_list.shape[0]):
		# print(keys[-1])
		ind = np.searchsorted(times[i], tr_list[t])
		# print(tr[t], w2v[ind,:], ind)
		if ind == 0:
			stim_run[t] = flat_ca[i][0]
		else:
			stim_run[t] = flat_ca[i][ind-1]
	print('stim shape: {0} for run {1}'.format(len(stim_run), i))
	stim.append(stim_run)

twosec= []
for i in range(1,5):
	vec = json.load(open('/idata/DBIC/cara/life/old_codes/Part{0}_Raw_Data.json'.format(i)))
	keys = []
	vals = []
	key_list = [int(k) for k in vec['features'].keys()]
	for key in sorted(key_list):
		keys.append(key)
		val = vec['features'][str(key)]
		# val = [_.strip().split(' ') for _ in val]
		# val = [item for sublist in val for item in sublist]
		vals.append(val)
	keys = np.array(keys)
	print(keys[-1] + 2)
	tr_list = np.arange(0, keys[-1] + 2, 2.5)
	stim_run = [[0 for x in range(1)] for x in range(tr_list.shape[0])]
	for t in range(tr_list.shape[0]):
		# print(keys[-1])
		ind = np.searchsorted(keys, tr_list[t])
		# print(tr[t], w2v[ind,:], ind)
		if ind == 0:
			stim_run[t] = vals[0]
		else:
			stim_run[t] = vals[ind-1]
	print('stim shape: {0} for run {1}'.format(len(stim_run), i))
	twosec.append(stim_run)


print('Done loading runfiles')
flat_tr = [item for sublist in stim for item in sublist]
actions = []

# trim the first 4 TRs for each fmri run
concat_stim = []
for i in range(len(fmri)):
	concat_stim.append(fmri[i][4:,:])

concat_stim = np.concatenate(concat_stim, axis=0)

cam_list.append(cam[:366,:])
cam_list.append(cam[366:704,:])
cam_list.append(cam[704:1073,:])
cam_list.append(cam[1073:,:])

for i in range(len(cam_list)):
	this = cam_list[i]
	cam_list[i] = np.concatenate((this[3:,:], this[2:-1,:], this[1:-2,:], this[:-3,:]), axis=1)

for i in range(len(fmri)):
	fmri[i] = fmri[i][4:,:]

for i in range(len(cam_list)):
	print(cam_list[i].shape)

stim = cam_list
resp =
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
								{'poison_dart_frog':'poison arrow		frog'}, {'dorsal_fin':'dorsal fin'}, {'':','}, {'chinstrap_penguin':'chinstrap penguin'}, \
								{'zoom':'zoomed in'},{'zoom':'zoomed out'}, {'blue_sky':'blue_sky'}, {'forked_tongue':'forked tongue'}, \
								{'rattlesnake':'rattle snake'}, {'areca_palm':'nut palm plant'}, {'tidepool':'tidal pool'}, \
								{'praying_mantis':'praying mantis'}, {'venus_flytrap':'venus fly trap'}, {'leopard_seal':'leopard seal'}, \
								{'spiderweb':'spider web'}, {'':' but'}, {'':'music'}, {'caging':'encaging'}, \
								{'seafloor':'sea_floor'}, {'silhouette':'silhoutte'}, {'swinging':'brachiating'}]

processed = TextLabeler(flat_tr, lod)
proc = [[words for segments in i for words in segments.split() if len(words)>2] for i in processed.text]

processed = TextLabeler(flat_ca, lod)
proc = [[words for segments in i for words in segments.split() if len(words)>2] for i in processed.text]

def create_w2v(stim):
	google_mean_vecs = []
	for i, tr in enumerate(stim):
		google_vec = []
		for word in tr:
			google_vec.append(google[word])

		goog_mean = np.mean(np.column_stack(google_vec), axis=1)[None, :]
		print(goog_mean.shape)
		google_mean_vecs.append(goog_mean)

	goog = np.concatenate(google_mean_vecs, axis=0)
	print('google: {0}'.format(goog.shape))

	return(goog)

goog = create_w2v(processed[0])
np.save('google_w2v_faces.npy', goog)

faces.csv
strict_animals.csv
loose_actions.csv
loose_places.csv
body_parts.csv
loose_animals.csv
loose_bg.csv
loose_objects.csv

actions = [item for sublist in proc for item in sublist if item in a]
objects = [item for sublist in stim for item in sublist if item in o]
animals = [item for sublist in stim for item in sublist if item in an]
places = [item for sublist in stim for item in sublist if item in p]

print(a)
print(o)



# loose and strict set
# full set
# strict set - actions, objects, places, animals, faces
# loose sets - actions, objects, places, animals, faces
# make phrases like "bird noise" to birdcall
a |= set(['dance', 'startled', 'ripped', 'spray', 'cracked', 'splash', 'smashed', 'eaten'])

actions = a
loose_objects = ['seaweed', 'circle', 'food', 'words', 'bubbles', 'clouds', 'drop', 'carcass', 'rocks', 'rock', \
								 'bark', 'flower', 'leaf', 'sun', 'shadow', 'spiderweb', 'stone', 'leaves', 'moss', \
								 'snakeskin', 'puddle', 'bubble', 'nut', 'stick', 'plants', 'word', 'peelings', 'mound', \
								 'sticks', 'ferns', 'nuts', 'branch', 'droplets', 'circles', 'plant', 'branches', 'Earth']
loose_places = ['underwater', 'hills', 'cliff', 'sky', 'silt', 'iceberg', 'trees', 'coast', 'sand', 'river', \
								 'steam', 'icebergs', 'prairie', 'sunrise', 'ice', 'mud', 'fog', 'seabed', 'tree_trunk', \
								 'water', 'shrubbery', 'dirt', 'rainforest', 'blue_sky', 'lake', 'seafloor', 'mist', 'landscape', 'tree_trunks', \
								 'muddy', 'mountains', 'sunlight', 'trail', 'reef',		'tree', 'den', 'lakebed', 'savanna', 'grass',		\
								 'reeds', 'flat', 'mountain', 'glacier', 'snow', 'field', 'forest', 'beach', 'arctic', 'waves', \
								 'wave', 'land', 'light', 'ocean', 'shore', 'lagoon', 'barren', 'wind', 'plains']

loose_animals = ['antelope', 'hippo', 'seals', 'birds', 'squirrels', 'tail', 'tusk', 'dragonfly', 'grebe', 'chameleon', \
								 'swordfish', 'eggs', 'tentacle', 'giraffe', 'tails',		'hands', 'foot', 'young', 'penguins', \
								 'fish', 'back', 'hair', 'dorsal_fin', 'zebra', 'seal', 'deer', 'hippos', 'killer_whale', 'rattlesnake', \
								 'tadpole', 'praying_mantis', 'bird', 'octopus', 'leg', 'heads', 'dorsal_fins', 'moths', 'ostrich', 'orangutan', \
								 'tongue', 'head', 'venus_flytrap', 'feet', 'eye', 'monkey', 'leopard_seal', 'tentacles', 'frog', \
								 'crab', 'legs', 'cricket', 'squirrel', 'tiger', 'baby', 'bug', 'penguin', 'lion', 'lizard', \
								 'dolphin', 'deer', 'eyes', 'ant', 'fly', 'face', 'capuchin', 'cheetah', 'mouths', \
								 'killer_whales', 'hand', 'insect', 'mouth', 'hummingbird', 'lion_cub', 'wings', 'monkeys', 'teeth', 'egg', 'whale']

faces = ['face', 'faces', 'eye', 'eyes', 'mouth', 'mouths', 'head', 'heads']
body_parts = ['face', 'faces', 'eye', 'eyes', 'mouth', 'mouths', 'tail', 'tusk', 'tentacle', 'tails', 'hands', 'hand', 'teeth', 'tooth', 'foot', 'back', 'hair', 'dorsal_fin', 'head', 'heads', 'leg', 'tongue', 'feet', 'tentacles', 'legs', 'dorsal_fins', 'wings']
unk = ['gray', 'dead', 'red', 'dry', 'fast', 'closed', 'open', 'pitted', 'green', 'left']

with open('loose_bg.csv', 'w') as f:
	wr = csv.writer(f, delimiter='\n')
	wr.writerow(list(loose_bg))

words_dir = '/ihome/cara/semantic_categories'

processed = []
for filename in os.listdir(words_dir):
	if '.csv' in filename:
		with open(os.path.join(words_dir, filename), 'r') as f:
			print(filename)
			keep = []
			re = csv.reader(f, delimiter='\n')
			for row in re:
				keep.extend(row)
			keep_set= set(keep)
			filtered = [[word if word in keep_set else '' for word in tr] for tr in proc for word in tr if word in keep_set]
			processed.append(filtered)

l = [[u'noise', u'chewing'],
 [u'tall', u'landscape', u'filled', u'rock'],
 [u'fade', u'out'],
 [u'water', u'noise'],
 [u'dolphin', u'noise'],
 [u'vocalization', u'deer'],
 [u'fish'],
 [u'leaves', u'dead'],
 [u'deer', u'chital'],
 [u'noise', u'squawking', u'penguin'],
 [u'vocalization', u'seal'],
 [u'vocalization', u'animal'],
 [u'noise', u'ostrich'],
 [u'egg', u'lays'],
 [u'trunks', u'tree'],
 [u'cheetah', u'breathing', u'noise'],
 [u'zoomed', u'out'],
 [u'taking', u'off'],
 [u'swimming'],
 [u'chinstrap', u'penguin'],
 [u'trail', u'silt'],
 [u'vocalization', u'squirrel'],
 [u'head', u'chameleon'],
 [u'whale', u'killer'],
 [u'flat', u',', u'but', u'landscape', u'rock', u'tall', u'filled'],
 [u'cheetah', u'head'],
 [u'footstep', u'noise'],
 [u'foot', u'lizard'],
 [u'pitted', u'rock'],
 [u'noise', u'seal'],
 [u'brown', u'tufted', u'capuchin'],
 [u'praying', u'mantis'],
 [u'noise', u'bubble'],
 [u'noise', u'forest'],
 [u'vocalizations', u'animal'],
 [u'fins', u'dorsal'],
 [u'pushing', u'dirt'],
 [u'shaking', u'leg'],
 [u'cooing', u'ostrich'],
 [u'crab-eater', u'seals'],
 [u'tidal', u'pool'],
 [u'noise', u'deer', u'chital'],
 [u'seaweed', u'reef'],
 [u'noise', u'eating'],
 [u'grooming', u'noise', u'crackling'],
 [u'noise', u'cracking'],
 [u'forked', u'tongue'],
 [u'arctic', u'landscape'],
 [u'tails', u'monkey'],
 [u'noise', u'rattlesnake', u'rattling'],
 [u'birds', u'background', u'in'],
 [u'noise', u'crackling'],
 [u'sizzling', u'sound'],
 [u'dry', u'lakebed'],
 [u'whale', u'noise'],
 [u'noise', u'flapping'],
 [u'rattle', u'noise', u'snake', u'rattling'],
 [u'breathing', u'noise'],
 [u'sound', u'dolphin', u'blowing'],
 [u'flying', u'dirt'],
 [u'flat,', u'but', u'filled', u'rock', u'tall', u'landscape'],
 [u'seals', u'elephant'],
 [u'tool', u'rock'],
 [u'tail', u'shaking'],
 [u'aerial', u'view'],
 [u'web', u'spider'],
 [u'cheetah', u'noise'],
 [u'blue', u'sky'],
 [u'noise', u'animal'],
 [u'nut', u'palm', u'plant'],
 [u'buffalo', u'carcass'],
 [u'zoomed', u'in'],
 [u'into', u'den', u'crawling'],
 [u'monkey', u'langur'],
 [u'noise', u'chirping'],
 [u'', u'frog', u'arrow', u'poison'],
 [u'tail', u'squirrel'],
 [u'fin', u'dorsal'],
 [u'monkey', u'face'],
 [u'leopard', u'seal'],
 [u'dear', u'chital'],
 [u'fish', u'flying'],
 [u'tail', u'snake'],
 [u'grebe'],
 [u'penguin'],
 [u'noise', u'ocean'],
 [u'tongue', u'noise'],
 [u'frog', u'arrow', u'poison'],
 [u'blowing', u'grass', u'noise'],
 [u'leaves', u'rustling'],
 [u'dried', u'lagoon'],
 [u'noise', u'static', u'vibrating'],
 [u'motion', u'slow'],
 [u'spouting', u'noise'],
 [u'noise', u'bird'],
 [u'mound', u'dirt'],
 [u'baby', u'orangutan'],
 [u'noise', u'banging'],
 [u'vocalization', u'monkey'],
 [u'leg', u'chameleon'],
 [u'hair', u'fine'],
 [u'sizzling', u'noise'],
 [u'vocalizations', u'monkey'],
 [u'penguin', u'dead', u'bloody'],
 [u'white', u'noise'],
 [u'barren', u'land'],
 [u'stick', u'bug'],
 [u'water', u'droplets'],
 [u'tree', u'trunk'],
 [u'noise', u'wind'],
 [u'vocalization', u'bird'],
 [u'noise', u'wing'],
 [u'fly', u'venus', u'trap'],
 [u'baby', u'penguin'],
 [u'vocalization', u'penguin'],
 [u'crab-eater', u'seal'],
 [u'buzzing', u'noise', u'bee'],
 [u'baby', u'octopus'],
 [u'running'],
 [u'black', u'fading'],
 [u'on', u'hippo', u'sitting'],
 [u'fading', u'out'],
 [u'sea_floor'],
 [u'motion', u'fast'],
 [u'squaking', u'noise', u'ostrich'],
 [u'display', u'ostrich'],
 [u'noises', u'seal'],
 [u'water'],
 [u'hippo', u'noise'],
 [u'sound', u'spraying'],
 [u'buzzing', u'noise'],
 [u'noise', u'rock'],
 [u'whales', u'killer'],
 [u'seal', u'elephant']]
# #
# # run 1
# unknown1 = ['red', 'fast', 'left']
#
# object1 = ['iceberg', 'bark', 'spiderweb', 'leaf', 'bubble', \
#								 'seaweed',		'circle', 'silt', 'branch', 'food', \
#								 'tree', 'word', 'cloud', 'rock', 'steam', \
#								 'flower', 'stick', 'snow', 'plant', 'wave']
#
# animal1 = ['antelope', 'seal', 'dragonfly', 'giraffe', 'wing', \
#								 'fish', 'dorsal_fin', 'zebra', 'seal', 'killer_whale', \
#								 'praying_mantis', 'bird', 'leg', 'ostrich', 'feet', \
#								 'whale', 'eye', 'monkey', 'frog', 'bug', \
#								 'lizard', 'dolphin', 'head', 'ant', 'cheetah', \
#								 'cricket', 'insect', 'chameleon', 'tongue', 'hummingbird', \
#								 'face', 'teeth']
#
# place1 = ['underwater', 'coast', 'seabed', \
#								 'rainforest', 'plains', 'mountains', 'trail', \
#								 'reef', 'savanna', 'ocean', 'light', \
#								 'forest', 'arctic', 'Earth', 'ice', 'water','grass']
#
# # run 2
# unknown2 = []
#
# object2 = ['rock', 'snakeskin', 'stick', 'tree', 'tree_trunk', \
#								 'branch', 'wave', 'seaweed', 'water', 'shrubbery', \
#								 'leaf', 'stick']
#
# animal2 = ['seal', 'bird', 'squirrel', 'tail', 'monkey', \
#								 'penguin', 'killer_whale', 'rattlesnake', 'dorsal_fin', \
#								 'tongue', 'feet', 'eye', 'leg', 'tiger', \
#								 'bug', 'penguin', 'deer', 'head', 'teeth', 'moth', \
#								 'tail', 'young']
#
# place2 = ['hill', 'sand', 'prairie', 'dirt', 'sunlight', \
#								 'mound', 'grass', 'forest', 'beach', \
#								 'ocean', 'shore', 'wind']
#
# # run 3
# unknown3 = ['carcass', 'dead', 'open', 'closed', \
#								 'pitted', 'fly', 'gray',		'green']
#
# object3 = ['bark', 'stone', 'leaf', 'tree', 'nut', \
#								 'plant', 'shadow', 'cloud', 'rock', 'peelings', \
#								 'tree_trunk', 'water', 'shrubbery', 'droplets', 'branch', ]
#
# animal3 = ['eye', 'bird', 'tusk', 'swordfish', 'hand', \
#								 'monkey', 'fish', 'hair', 'bird', 'leg', \
#								 'insect', 'venus_flytrap', 'head', 'lion', 'hippo', \
#								 'lion_cub', 'capuchin', 'mouth']
#
# place3 = ['underwater', 'river', 'sunrise', \
#								 'sun', 'mud', 'landscape', 'dirt', 'blue_sky', 'muddy', \
#								 'plains', 'flat', 'dry', 'mountains', 'field', \
#								 'forest', 'cliff', 'land', 'lakebed', 'ocean', 'shore', \
#								 'lagoon', 'barren']
# # run 4
# unknown4 = ['drop', 'back', 'dead']
#
# object4 = ['tree_trunk', 'rock', 'leaf', 'tree', \
#								 'plant', 'wave', 'branch', 'stick', 'seaweed', \
#								 'shadow', 'puddle']
#
# animal4 = ['bird', 'tail', 'tentacle', 'foot', 'tadpole', \
#								 'grebe', 'leg', 'head', 'egg', 'hand', \
#								 'orangutan', 'octopus', 'feet', 'eye', 'frog', \
#								 'leopard_seal', 'crab', 'fish', 'baby', 'penguin']
#
# place4 = ['underwater', 'sky', 'lake', 'cloud', 'river', \
#								 'sun', 'ice', 'water', 'seafloor', 'rainforest', 'mist', 'mountains' \
#								 'iceberg', 'sticks', 'reeds', 'ferns', 'glacier', \
#								 'nuts', 'forest', 'branch', 'moss', 'cliff', \
#								 'land', 'ocean', 'shore','fog', 'den']
#
#
# action1 = a[0] + ['spray', 'splash']
# action2 = a[1] + ['startled']
# action3 = a[2] + ['smashed', 'cracked']
# action4 = a[3] + ['dance', 'splash', 'ripped', 'leaves', 'eaten']
