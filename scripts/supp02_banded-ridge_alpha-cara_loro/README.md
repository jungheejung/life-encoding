## Banded Ridge

### Where is this code from?
* We carefully followed the banded ridge regression tutorial from the Gallant lab ([link](https://nbviewer.jupyter.org/github/gallantlab/tikreg/blob/master/examples/tutorial_banded_ridge_polar.ipynb)).

### What is being done in this code?
* In this code, we use "background", "agents", and "actions" as our feature. 
* In our code, we use Cara's alphas and ratios (whereas the notebook tutorial walksthrough one or a number of alphas)

### Are there any changes to the original tikreg code?
* instead of searching the entire alpha space, we use one alpha -- 18.33 -- from Van Uden (2018)
* The main analysis will search for the entire space.

### How do I use this code on Discovery? (or any PBS scheduling system)
* submit the jobs via `mksub BANDEDRIDGE01_submit.sh` on your HPC. 
* Jobs are submitted as job arrays, pulled from the canonical_sublist.txt
* In other words, each hemisphere, participant, feature is submitted as a separate job

---

## ISC
ISC

--- 

## Average correlation coefficient between expected Y and beta weights
THREE
