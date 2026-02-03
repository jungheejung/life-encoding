# load json
# per sentence, 
# %%
import json
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
# %%
json_fpath = '/Users/h/Documents/projects_local/life-encoding/scripts/revision/annotation_output_2/video_annotations_ses-01_run-01_order-02_content-wanderers_198.20s.json'
with open(json_fpath, 'r') as f:
    annot_json = json.load(f)

# %%
annotations = annot_json['annotations']

# %%
all_embeddings = []
for entry in annotations:
    timestamp = entry[0]  # TR
    word_lists = entry[1:]  # lists of words
    
    print(f"Processing timestamp: {timestamp}")

    entry_embeddings = []
    for i, word_list in enumerate(word_lists):
        # print(word_list)
        text = ' '.join(word_list)
        # option 2: extract embeddings for each word and average them. 
        # TODO: for word in word_list:
        # print(text)
        # average the embeddings later
        embedding = model.encode(text, show_progress_bar=True)
        entry_embeddings.append(embedding)
        print(f"{i} {word_list} {text} -> embedding shape: {embedding.shape}")

    # Store the embeddings for this timestamp
    all_embeddings.append({
        'timestamp': timestamp,
        'embeddings': entry_embeddings
    })