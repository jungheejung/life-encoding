import subprocess as sp
import gensim
#from gensim.models import word2vec

f = file('part1_words_edit.tsv', 'r')
g = file('part2_words_edit.tsv', 'r')
h = file('part3_words_edit.tsv', 'r')
i = file('part4_words_edit.tsv', 'r')

fn = 'GoogleNews-vectors-negative300.bin.gz'
model = gensim.models.KeyedVectors.load_word2vec_format(fn,binary=True)
#model = word2vec.KeyedVectors.load_word2vec_format(f,binary=True)

d = {}
for line in f:
    (key) = line.strip()
    d[key] = model.word_vec(key)
    #word = x.split()
    #print word[0]
    #model.word_vec(word)
    #print word
with open ('mydict.json', 'w') as f:
    for key, value in d.items():
        f.write('%s\n%s\n' % (key,value))

"""
part2 = {}
for line in g:
    (key) = line.strip()
    part2[key] = model.word_vec(key)
with open ('mydict2.json', 'w') as g:
    for key, value in part2.items():
        g.write('%s\n%s\n' % (key,value))

part3 = {}
for line in h:
    (key) = line.strip()
    part3[key] = model.word_vec(key)
with open ('mydict3.json', 'w') as h:
    for key, value in part3.items():
        h.write('%s\n%s\n' % (key,value))

part4 = {}
for line in i:
    (key) = line.strip()
    part4[key] = model.word_vec(key)
with open ('mydict4.json', 'w') as i:
    for key, value in part4.items():
        i.write('%s\n%s\n' % (key,value))

"""






