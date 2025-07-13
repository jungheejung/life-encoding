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
# https://huggingface.co/datasets/merve/coco/blob/main/annotations/captions_train2017.json

an_captions_f = 'captions_activitynet.json'
# https://cs.stanford.edu/people/ranjaykrishna/densevid/

output_f = 'agent_words.txt'
top_k = 1000

agent_keywords = ('person.n.', 'animal.n.', 'people.n',
                  'body_covering.n', 'occupation.n')

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
def is_animate_noun(word):
    for syn in wn.synsets(word, pos=wn.NOUN):
        for path in syn.hypernym_paths():
            for h in path:
                #if h.name().startswith(("person.n.", "animal.n.",
                #"organism.n.", "living_thing.n.", "casual_agent.n.")):
                if h.name().startswith(agent_keywords):
                    return True
    return False

# Filter for animate words in wordnet
def is_animate_noun(word):
    count_agent_syns = 0
    syns = wn.synsets(word, pos=wn.NOUN)
    if len(syns) == 0:
        return False
    for syn in syns:
        paths = syn.hypernym_paths()
        search_agent_path = True
        while search_agent_path:
            for path in paths:
                for h in path:
                    #if h.name().startswith(("person.n.", "animal.n.",
                    #"organism.n.", "living_thing.n.")):
                    if (h.name().startswith(agent_keywords)
                        and search_agent_path):
                        count_agent_syns += 1
                        search_agent_path = False
                        break
            search_agent_path = False
    assert count_agent_syns <= len(syns)
    if count_agent_syns >= (len(syns) * .9):
        return True
    return False

# Filter for animate words in wordnet
def is_animate_adj(word):
    count_agent_syns = 0
    syns = wn.synsets(word, pos=wn.NOUN)
    for syn in syns:
        paths = syn.hypernym_paths()
        search_agent_path = True
        while search_agent_path:
            for path in paths:
                for h in path:
                    if (h.name().startswith(('person.n.', 'people.n', 'animal.n.', 'body_covering.n.01'))
                        and search_agent_path):
                        count_agent_syns += 1
                        search_agent_path = False
                        break
            search_agent_path = False
    assert count_agent_syns <= len(syns)
    if count_agent_syns >= (len(syns) * .9):
        return True
    return False

# Exclude stopwords and grab nouns/adjectives
def is_agent_descriptor(token):
    if token.is_stop or not re.match(r"^[a-z]+$", token.text):
        return False
    # Keep adjectives (describing traits) and animate nouns (e.g., man, dog)
    #if token.pos_ == "ADJ" and is_animate_adj(token.lemma_):
    #    return True
    #if token.pos_ == "ADJ":
    #    return True
    if token.pos_ == 'NOUN' and is_animate_noun(token.lemma_):
        return True
    return False

# Run word exctraction
def extract_agents(captions):
    counter = Counter()
    print("Processing captions...")
    for caption in tqdm(captions):
        doc = nlp(caption)
        for token in doc:
            if is_agent_descriptor(token):
                counter[token.lemma_] += 1
    return counter

# Save output to a filename
def save_results(counter, top_k, output_path):
    top_words = counter.most_common(top_k)
    with open(output_path, "w") as f:
        for word, freq in top_words:
            f.write(f"{word}\t{freq}\n")
    print(f"\nTop {top_k} agent descriptors saved to '{output_path}'")
    print("Top 20 examples:")
    for word, freq in top_words[:20]:
        print(f"{word}: {freq}")

# Run the script
def main():
    captions = load_coco(coco_captions_f)
    captions += load_activitynet(an_captions_f)
    counter = extract_agents(captions)
    save_results(counter, top_k, output_f)

if __name__ == "__main__":
    main()
