import json
import spacy
import nltk
from nltk.corpus import wordnet as wn
from collections import Counter
from tqdm import tqdm
import re

# Load models
nlp = spacy.load('en_core_web_sm')
nltk.download('wordnet')

# Configure paths and filenames
coco_captions_f = 'captions_train2017.json'
an_captions_f = 'captions_activitynet.json'
places365_f = 'categories_places365.txt'
places_adj_f = 'adjectives_places365.csv'
output_f = 'scene_words.txt'
top_k = 1000

with open(places365_f) as f:
    places_words = f.readlines()
places_words = [w.split('/')[2].split()[0]
                for w in places_words]
n_places = len(places_words)
              
scene_keywords = ('environment.n.01', 'area.n.05', 'room.n.01',
                  'location.n.01', 'region.n.03',
                  'geographical_area.n.01', 'land.n.04',
                  'structure.n.01', 'establishment.n.04')

with open(places_adj_f) as f:
    scene_adjectives = f.readlines()
scene_adjectives = [w.strip() for w in scene_adjectives]

# Load captions
def load_coco(path):
    with open(path, 'r') as f:
        coco = json.load(f)
    return [ann['caption'].lower() for ann in coco['annotations']]

# Load captions
def load_activitynet(path):
    with open(path, "r") as f:
        an = json.load(f)
    sentences = [an[k]['sentences'] for k in an]
    return [s.lower().strip() for l in sentences for s in l]

# Filter for animate words in wordnet
def is_scene_noun(word):
    count_scene_syns = 0
    syns = wn.synsets(word, pos=wn.NOUN)
    if len(syns) == 0:
        return False
    for syn in syns:
        paths = syn.hypernym_paths()
        search_scene_path = True
        while search_scene_path:
            for path in paths:
                for h in path:
                    #if h.name().startswith(("person.n.", "animal.n.",
                    #"organism.n.", "living_thing.n.")):
                    if (h.name().startswith(scene_keywords)
                        and search_scene_path):
                        count_scene_syns += 1
                        search_scene_path = False
                        break
            search_scene_path = False
    assert count_scene_syns <= len(syns)
    if count_scene_syns >= (len(syns) * .9):
        return True
    return False

# Exclude stopwords and grab nouns/adjectives
def is_scene_descriptor(token):
    if token.is_stop or not re.match(r"^[a-z]+$", token.text):
        return False
    #if token.pos_ == "ADJ" and token.lemma_ in scene_adjectives:
    #    return True
    #if (token.pos_ == 'NOUN' and is_scene_noun(token.lemma_)
    #    and token.lemma_ not in places_words):
    #    return True
    if token.pos_ == 'NOUN' and is_scene_noun(token.lemma_):
        return True
    return False

# Run word exctraction
def extract_scene(captions):
    counter = Counter()
    print("Processing captions...")
    for caption in tqdm(captions):
        doc = nlp(caption)
        for token in doc:
            if is_scene_descriptor(token):
                counter[token.lemma_] += 1
    return counter

# Save output to a filename
def save_results(counter, top_k, output_path):
    top_words = counter.most_common(top_k)
    top_word_list = [w[0] for w in top_words]
    missing_places = []
    for places_word in places_words:
        if places_word not in top_word_list:
            missing_places.append((places_word, 1))
    if len(top_words) > (top_k - len(missing_places)):
        top_words = top_words[-len(missing_places):] + missing_places
    else:
        top_words += missing_places
    with open(output_path, "w") as f:
        for word, freq in top_words:
            f.write(f"{word}\t{freq}\n")
    print(f"\nTop {top_k} scene descriptors saved to '{output_path}'")
    print("Top 20 examples:")
    for word, freq in top_words[:20]:
        print(f"{word}: {freq}")

# Run the script
def main():
    captions = load_coco(coco_captions_f)
    captions += load_activitynet(an_captions_f)
    counter = extract_scene(captions)
    save_results(counter, top_k, output_f)

if __name__ == "__main__":
    main()
