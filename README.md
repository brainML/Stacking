# 🧠 Stacked regressions and structured variance partitioning for interpretable brain maps 

## Overview

This is a Python package that provides an implementation of stacked regression for functional MRI (fMRI) data. The package uses ridge regression to train models on multiple feature spaces and combines the predictions from these models using a weighted linear combination. The weights are learned using quadratic programming.

> Here we presents an approach for brain mapping based on two proposed methods: stacking different encoding models and structured variance partitioning. This package is useful for researchers interested in aligning brain activity with different layers of a neural network, or with other types of correlated feature spaces.

> Relating brain activity associated with a complex stimulus to different attributes of that stimulus is a powerful approach for constructing functional brain maps. However, when stimuli are naturalistic, their attributes are often correlated. These different attributes can act as confounders for each other and complicate the interpretability of brain maps. Correlations between attributes also impact the robustness of statistical estimators.

> Each encoding model uses as input a feature space that describes a different stimulus attribute. The algorithm learns to predict the activity of a voxel as a linear combination of the individual encoding models. We show that the resulting unified model can predict held-out brain activity better or at least as well as the individual encoding models. Further, the weights of the linear combination are readily interpretable; they show the importance of each feature space for predicting a voxel.

> We build on our stacking models to introduce a new variant of variance partitioning in which we rely on the known relationships between features during hypothesis testing. This approach, which we term structured variance partitioning, constraints the size of the hypothesis space and allows us to ask targeted questions about the similarity between feature spaces and brain regions even in the presence of correlations between the feature spaces.

> We validate our approach in simulation, showcase its brain mapping potential on fMRI data, and release a Python package.

## Installation
To use this code, you will need to have `numpy`, `cvxopt`, and `scipy` installed. You can install these packages using pip:


```bash
pip install numpy cvxopt scipy scikit-learn
```


## Usage
Here is an example of how to use the `stacking_fmri` function:
```python
from stacking_fmri import stacking_fmri
from sklearn.datasets import make_regression

# Generate synthetic data
X_train, y_train = make_regression(n_samples=50, n_features=1000, random_state=42)
X_test, y_test = make_regression(n_samples=50, n_features=1000, random_state=43)

# Generate random feature spaces
n_features = 5
train_features = [np.random.randn(X_train.shape[0], 10) for _ in range(n_features)]
test_features = [np.random.randn(X_test.shape[0], 10) for _ in range(n_features)]

# Train and test the model
(
    r2s,
    stacked_r2s,
    r2s_weighted,
    r2s_train,
    stacked_train_r2s,
    S,
) = stacking_fmri(
    X_train,
    X_test,
    train_features,
    test_features,
    method="cross_val_ridge",
    score_f=np.mean_squared_error,
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


## Contact
Created by [@lrg1213] - feel free to contact me!


## References
<a id="1">[1]</a> 
Ruogu Lin, Thomas Naselaris, Kendrick Kay, and Leila Wehbe (2023). 
Stacked regressions and structured variance partitioning for interpretable brain maps.



