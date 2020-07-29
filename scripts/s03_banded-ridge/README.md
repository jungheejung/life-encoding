## Banded Ridge

You will submit the jobs by typing `./c01_submit.sh` on your HPC. 
This will instigate `mksub` within `c02_submitpython.sh` and will now submit parallel jobs. 

We carefully followed the banded ridge regression tutorial from the Gallant lab ([link](https://nbviewer.jupyter.org/github/gallantlab/tikreg/blob/master/examples/tutorial_banded_ridge_polar.ipynb)).

In this code, we use "background" and "actions" as our feature. 
In our code, we use all of the alphas and ratios (whereas the notebook tutorial walksthrough one or a number of alphas)
```
X1_prior = spatial_priors.SphericalPrior(X1, hyparams=[lambda_one])
X2_prior = spatial_priors.SphericalPrior(X2, hyparams=[lambda_two])

CHANGED TO

X1_prior = spatial_priors.SphericalPrior(X1, hyparams=ratios)
X2_prior = spatial_priors.SphericalPrior(X2, hyparams=ratios)

```
