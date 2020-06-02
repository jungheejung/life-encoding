# life-encoding
This repo contains code for evaluating forward encoding models of animals, actions, and scenes using fMRI data collected while participants viewed the Life nature documentary

* google docs ["life encoding project"](https://docs.google.com/document/d/1kM9YBm-OaNljDIAbFfehQLxfugjoejJuO-nBYPkV1RU/edit?ts=5ec57676)
* Cara's "data": `/idata/DBIC/cara/life/`
* Cara's "code": `/ihome/cara/frontiers_18w/life/forward_encoding/`

## Data analysis procedures
### 1. Anatomical alignment and Hyperalignment
#### Scripts
* Anatomical
* Hyperalignment
#### Data
* Anatomical data:
* Hyperalignment data: 


### 2. Semantic Features
#### Scripts
* word2vector
#### Data
* Behavior
* Taxonomy
* Scene

### 3. Regularized Regression
#### Scripts
* Behavior
* Taxonomy
* Scene
* Behavior & Taxonomy
* Behavior & Scene
* Taxonomy & Scene
* Behavior & Taxonomy & Scene
#### Data
* Behavior
* Taxonomy
* Scene
* Behavior & Taxonomy
* Behavior & Scene
* Taxonomy & Scene
* Behavior & Taxonomy & Scene

### 4. Variance Partition
#### Scripts
* Behavior
* Taxonomy
* Scene
* Behavior & Taxonomy
* Behavior & Scene
* Taxonomy & Scene
* Behavior & Taxonomy & Scene
#### Data
* Behavior
* Taxonomy
* Scene
* Behavior & Taxonomy
* Behavior & Scene
* Taxonomy & Scene
* Behavior & Taxonomy & Scene

# Actual data structure in Cara's repository
```
idata/DBIC/cara/life
├──  data
│     ├── annotations 
│     │    └── 4 files: annotations_1.csv
│     └── audio 
│          └── 4 files: annotations_1.csv
│     
├──  pymvpa
│     ├── search_hyper_mappers_life_mask_nofsel_lh_leftout_1.hdf5
│     ├── ...
│     └── search_hyper_mappers_life_mask_nofsel_rh_leftout_4.hdf5
│
├──  ridge
│     ├── avg_corrs (e.g. annotations_1.csv)
│     ├── models (e.g. life_part1.wav)
│     ├── incomplete
│     ├── niml
│     └── only actions.npy     only_animals.npy               only_bg.npy            
│     
└──  semantic_cat
      ├── actions_bg.npy       bg_animals.npy                 faces.npy
      ├── actions.npy          bg.npy                         objects.npy
      ├── all.npy              body_parts.npy                 places.npy
      ├── animals_actions.npy  body_parts_strict_animals.npy  strict_animals_faces.npy
      └── animals.npy          faces_body_parts.npy           strict_animals.npy
```
