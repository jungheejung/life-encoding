import json
import spacy
import nltk
from nltk.corpus import wordnet as wn
from collections import Counter
from tqdm import tqdm
import re
from os.path import exists
from pyinflect import getInflection


# Load NLP models
nlp = spacy.load('en_core_web_sm')
nltk.download('wordnet')

# Configure paths and filenames
coco_captions_f = 'captions_train2017.json'
an_captions_f = 'captions_activitynet.json'
output_f = 'action_words.txt'
top_k = 1000

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

# Filter for verb/action words
def is_action_word(token):
    if token.is_stop or not re.match(r"^[a-z]+$", token.text):
        return False
    if (token.pos_ in {'VERB', 'AUX'}
        and len(wn.synsets(token.lemma_, pos=wn.VERB)) > 0):
        return True
    #if token.pos_ in {'VERB', 'AUX', 'ADV'}:
    #    return True
    # Optionally include nouns that describe actions (e.g., "jump", "swim") used as gerunds
    #if token.pos_ == "NOUN":
    #    synsets = wn.synsets(token.lemma_, pos=wn.NOUN)
    #    for syn in synsets:
    #        for path in syn.hypernym_paths():
    #            for h in path:
    #                if h.name().startswith("act.n.") or h.name().startswith("event.n.") or h.name().startswith("process.n."):
    #                    return True
    #return False

# Run word extraction
def extract_actions(captions):
    counter = Counter()
    print("Processing captions...")
    for caption in tqdm(captions):
        doc = nlp(caption)
        for token in doc:
            if is_action_word(token):
                counter[token.lemma_] += 1
    return counter

# Save output to a filename
def save_results(counter, top_k, output_path):
    top_words = counter.most_common(top_k)
    with open(output_path, "w") as f:
        for word, freq in top_words:
            f.write(f"{word}\t{freq}\n")
    print(f"\nTop {top_k} action descriptors saved to '{output_path}'")
    print("Top 20 examples:")
    for word, freq in top_words[:20]:
        print(f"{word}: {freq}")

# Run the script
def main():
    captions = load_coco(coco_captions_f)
    captions += load_activitynet(an_captions_f)
    counter = extract_actions(captions)
    save_results(counter, top_k, output_f)
    
if __name__ == "__main__":
    main()


if exists(output_f):
    with open(output_f) as f:
        output = f.readlines()
    
    gerunds = []
    for verb in output:
        freq = verb.split('\t')[1].strip()
        g = getInflection(verb.split('\t')[0], 'VBG')
        if not g:
            gerunds.append(verb)
            print(f'No gerund for {verb}')
        else:
            gerunds.append(g[0] + '\t' + freq + '\n')
    
    gerund_f = 'action_words_gerunds.txt'
    with open(gerund_f, 'w') as f:
        f.writelines(gerunds)
        