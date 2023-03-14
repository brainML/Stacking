# Stacked regressions and structured variance partitioning for interpretable brain maps

## Overview
> This Python package presents an approach for brain mapping based on two proposed methods: stacking different encoding models and structured variance partitioning. This package is useful for researchers interested in aligning brain activity with different layers of a neural network, or with other types of correlated feature spaces.

> Relating brain activity associated with a complex stimulus to different attributes of that stimulus is a powerful approach for constructing functional brain maps. However, when stimuli are naturalistic, their attributes are often correlated. These different attributes can act as confounders for each other and complicate the interpretability of brain maps. Correlations between attributes also impact the robustness of statistical estimators.

> Each encoding model uses as input a feature space that describes a different stimulus attribute. The algorithm learns to predict the activity of a voxel as a linear combination of the individual encoding models. We show that the resulting unified model can predict held-out brain activity better or at least as well as the individual encoding models. Further, the weights of the linear combination are readily interpretable; they show the importance of each feature space for predicting a voxel.

> We build on our stacking models to introduce a new variant of variance partitioning in which we rely on the known relationships between features during hypothesis testing. This approach, which we term structured variance partitioning, constraints the size of the hypothesis space and allows us to ask targeted questions about the similarity between feature spaces and brain regions even in the presence of correlations between the feature spaces.

> We validate our approach in simulation, showcase its brain mapping potential on fMRI data, and release a Python package.


## Usage
We provide examples of how to use the package in jupyter notebooks:

- stacking_tutorial.ipynb

-variance_partitioning.ipynb


## Project Status
Project is: _complete_ 


## Contributions
We welcome contributions to this package. If you find any bugs or have suggestions for improvements, please open an issue on our GitHub repository.


## Contact
Created by [@lrg1213] - feel free to contact me!
