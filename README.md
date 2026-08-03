# 🧠 Stacked regressions and structured variance partitioning for interpretable brain maps 

## Overview

This project provides an implementation of stacked regression for functional
MRI (fMRI) data. It trains ridge-regression models on multiple feature spaces
and combines their predictions with weights learned by quadratic programming.

The code supports brain mapping with stacked encoding models and structured
variance partitioning. It can be used with neural-network representations or
other correlated feature spaces.

> Relating brain activity associated with a complex stimulus to different attributes of that stimulus is a powerful approach for constructing functional brain maps. However, when stimuli are naturalistic, their attributes are often correlated. These different attributes can act as confounders for each other and complicate the interpretability of brain maps. Correlations between attributes also impact the robustness of statistical estimators.

> Each encoding model uses as input a feature space that describes a different stimulus attribute. The algorithm learns to predict the activity of a voxel as a linear combination of the individual encoding models. We show that the resulting unified model can predict held-out brain activity better or at least as well as the individual encoding models. Further, the weights of the linear combination are readily interpretable; they show the importance of each feature space for predicting a voxel.

Structured variance partitioning uses known relationships between features to
constrain the hypothesis space and support targeted comparisons between feature
spaces and brain regions.

> We validate our approach in simulation, showcase its brain mapping potential on fMRI data, and release a Python package.

## Installation
Install the runtime dependencies, then run the examples from the repository
root:


```bash
python -m pip install numpy scipy scikit-learn
```


## Usage
Here is a self-contained example using `stacking_fmri`:
```python
import numpy as np

from stacking import stacking_fmri

# Generate synthetic response matrices and feature spaces.
rng = np.random.default_rng(42)
n_train, n_test, n_targets = 50, 20, 12
train_data = rng.normal(size=(n_train, n_targets))
test_data = rng.normal(size=(n_test, n_targets))

n_features = 5
train_features = [rng.normal(size=(n_train, 10)) for _ in range(n_features)]
test_features = [rng.normal(size=(n_test, 10)) for _ in range(n_features)]

# Train and test the model
(
    r2s,
    stacked_r2s,
    r2s_weighted,
    r2s_train,
    stacked_train_r2s,
    S,
) = stacking_fmri(
    train_data,
    test_data,
    train_features,
    test_features,
    method="cross_val_ridge",
)

print("R2 scores for each feature and voxel:")
print(r2s)
print("\nWeighted R2 scores for each feature and voxel:")
print(r2s_weighted)
print("\nUnweighted R2 scores for the stacked predictions:")
print(stacked_r2s)
print("\nStacking weights:")
print(S)
```

We also provide examples of how to use the package in jupyter notebooks:

- stacking_tutorial.ipynb

- variance_partitioning.ipynb


<!-- ## Project Status
Project is: _complete_  -->


## Contributions
Contributions are welcome! Please feel free to submit a pull request with your changes or open an issue to report a bug or suggest a new feature.


## References
Ruogu Lin, Thomas Naselaris, Kendrick Kay, and Leila Wehbe (2023).
*Stacked regressions and structured variance partitioning for interpretable
brain maps*.


