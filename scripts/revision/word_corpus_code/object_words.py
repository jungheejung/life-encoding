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
output_f = 'object_words.txt'
top_k = 1000
              
object_keywords = ('device.n', 'instrumentality.n', 'tool.n',
                   'food.n', 'furniture.n', 'plant_part.n',
                   'covering.n')

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
def is_object_noun(word):
    count_object_syns = 0
    syns = wn.synsets(word, pos=wn.NOUN)
    if len(syns) == 0:
        return False
    for syn in syns:
        paths = syn.hypernym_paths()
        search_object_path = True
        while search_object_path:
            for path in paths:
                for h in path:
                    #if h.name().startswith(("person.n.", "animal.n.",
                    #"organism.n.", "living_thing.n.")):
                    if (h.name().startswith(object_keywords)
                        and search_object_path):
                        count_object_syns += 1
                        search_object_path = False
                        break
            search_object_path = False
    assert count_object_syns <= len(syns)
    if count_object_syns >= (len(syns) * .9):
        return True
    return False

# Exclude stopwords and grab nouns/adjectives
def is_object_descriptor(token):
    if token.is_stop or not re.match(r"^[a-z]+$", token.text):
        return False
    if token.pos_ == 'NOUN' and is_object_noun(token.lemma_):
        return True
    return False

# Run word exctraction
def extract_object(captions):
    counter = Counter()
    print("Processing captions...")
    for caption in tqdm(captions):
        doc = nlp(caption)
        for token in doc:
            if is_object_descriptor(token):
                counter[token.lemma_] += 1
    return counter

# Save output to a filename
def save_results(counter, top_k, output_path):
    top_words = counter.most_common(top_k)
    with open(output_path, "w") as f:
        for word, freq in top_words:
            f.write(f"{word}\t{freq}\n")
    print(f"\nTop {top_k} object descriptors saved to '{output_path}'")
    print("Top 20 examples:")
    for word, freq in top_words[:20]:
        print(f"{word}: {freq}")

# Run the script
def main():
    captions = load_coco(coco_captions_f)
    captions += load_activitynet(an_captions_f)
    counter = extract_object(captions)
    save_results(counter, top_k, output_f)

if __name__ == "__main__":
    main()
