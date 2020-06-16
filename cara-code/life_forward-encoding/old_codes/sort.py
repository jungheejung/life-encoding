import pickle
i = range(4)
f = pickle.load(file('part1_raw_data.pkl', 'r')) 

features = f['features']
all_features = f['all_features']
#m = file('part1_raw_data.tsv', 'w')
#keys = features.keys()
#values = features.values()
fp = open('part1_raw_data.tsv', 'w')
f2 = open('part1_allwords.tsv', 'w')
for k,v in features.items():
    fp.write('%s\t%s\n' %  (k,v))
for x in all_features:
    f2.write('%s\n' % (x))

g = pickle.load(file('part2_raw_data.pkl', 'r')) 
features = g['features']
all_features = g['all_features']
fp = open('part2_raw_data.tsv', 'w')
f2 = open('part2_allwords.tsv', 'w')
for k,v in features.items():
    fp.write('%s\t%s\n' %  (k,v))
for x in all_features:
    f2.write('%s\n' % (x))

h = pickle.load(file('part3_raw_data.pkl', 'r')) 
all_features = h['all_features']
features = h['features']
fp = open('part3_raw_data.tsv', 'w')
f2 = open('part3_allwords.tsv', 'w')
for k,v in features.items():
    fp.write('%s\t%s\n' %  (k,v))
for x in all_features:
    f2.write('%s\n' % (x))

i = pickle.load(file('part4_raw_data.pkl', 'r'))
all_features = i['all_features']
features = i['features']
fp = open('part4_raw_data.tsv', 'w')
f2 = open('part4_allwords.tsv', 'w')
for k,v in features.items():
    fp.write('%s\t%s\n' %  (k,v))
for x in all_features:
    f2.write('%s\n' % (x))







"""
for line in str(keys):
    h = line.split(',')

    m = file('part1_raw_data.tsv', 'a')
    m.write('%s\n' % (keys))
print keys[0]
for line in str(keys):
    x = line.strip()
    col = x.split()
for line in str(values):
    x = line.strip()
    col = x.split()
    m.write('%s\t%s\n' % (keys, values))
print values.keys()

"""
