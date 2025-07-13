from itertools import combinations

places365_f = 'categories_places365.txt'
with open(places365_f) as f:
    places_words = f.readlines()
places365 = [w.split('/')[2].split()[0]
             for w in places_words]

categories = ['action', 'agent', 'scene', 'object']

word_tabs, word_lists, word_freqs, word_sets = {}, {}, {}, {}
for category in categories:
    with open(f'{category}_words.txt') as f:
        words_f = f.readlines()
    word_tabs[category] = words_f
    words = [w.split('\t')[0] for w in words_f]
    freqs = [int(w.split('\t')[1].strip()) for w in words_f]
    word_lists[category] = words
    word_freqs[category] = freqs
    word_sets[category] = set(words)
    
cat_pairs = list(combinations(categories, 2))

words_removed = {c: [] for c in categories}
for c1, c2 in cat_pairs:
    s1, s2 = word_sets[c1], word_sets[c2]
    overlaps = list(s1.intersection(s2))
    for overlap in overlaps:
        s1_id = word_lists[c1].index(overlap)
        s2_id = word_lists[c2].index(overlap)
        s1_freq = word_freqs[c1][s1_id]
        s2_freq = word_freqs[c2][s2_id]
        
        if c1 == 'scene' and overlap in places365:
            word_tabs[c2][s2_id] = None
            words_removed[c2].append(overlap)
        elif c2 == 'scene' and overlap in places365:
            word_tabs[c1][s1_id] = None
            words_removed[c1].append(overlap)
        elif s1_freq > s2_freq:
            word_tabs[c2][s2_id] = None
            words_removed[c2].append(overlap)
        elif s1_freq < s2_freq:
            word_tabs[c1][s1_id] = None
            words_removed[c1].append(overlap)
        else:
            print('Colliding word frequencies!')
            print(f'Removing {overlap} from '
                  f'both {c1} and {c2} lists')
            word_tabs[c1][s1_id] = None
            word_tabs[c2][s2_id] = None
            words_removed[c1].append(overlap)
            words_removed[c2].append(overlap)

with open('action_words_gerunds.txt') as f:
    gerunds = f.readlines()
assert len(word_tabs['action']) == len(gerunds)

for category in categories:
    word_tabs[category] = [w for w in word_tabs[category]
                           if w != None]
    print(f'{category}: {len(word_tabs[category])}')
    with open(f'{category}_words_trim.txt', 'w') as f:
        f.writelines(word_tabs[category])

gerunds = [g for a, g in zip(word_tabs['action'], gerunds)
           if a != None]
assert len(word_tabs['action']) == len(gerunds)
with open(f'action_words_gerunds_trim.txt', 'w') as f:
    f.writelines(gerunds)

# Remaining number of words after removing overlaps
# action: 959
# agent: 974
# scene: 712
# object: 968