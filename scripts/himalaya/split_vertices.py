import numpy as np
import json

# Create a dictionary of vertex splits
n_vertices = 40962
n_splits = 40

splits = {i: s for i, s in
          enumerate(np.array_split(np.arange(n_vertices), n_splits))}

assert len(splits) == n_splits
assert np.sum([s.shape for s in splits.values()])

print(f"Split vertices into {len(splits)} subsets of mean "
      f"length {np.mean([s.shape for s in splits.values()])}")

with open('rois.json', 'r') as f:
    rois = json.load(f)
    
for hemi in ['lh', 'rh']:
    if hemi not in rois:
        rois[hemi] = {}
    for i in splits:
        rois[hemi][str(i)] = splits[i].tolist()

with open('rois.json', 'w') as f:
    json.dump(rois, f)