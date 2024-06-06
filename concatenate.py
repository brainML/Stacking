# from cvxopt import matrix, solvers
import numpy as np
from scipy.stats import zscore
from ridge_tools import cross_val_ridge, R2, ridge
from stacking import feat_ridge_CV, get_cv_indices


def concatenate_CV_fmri(data, features, method="cross_val_ridge", n_folds=5, score_f=R2):
    """
    A function that concatenates feature spaces to predict fMRI signal.

    Args:
    - data (ndarray): A matrix of fMRI signal data with dimensions n_time x n_voxels.
    - features (list): A list of length n_features containing arrays of predictors
      with dimensions n_time x n_dim.
    - method (str): A string indicating the method to use to train the model. Default is "cross_val_ridge".
    - n_folds (int): An integer indicating the number of cross-validation folds to use. Default is 5.
    - score_f (function): A function to use for scoring the model. Default is R2.

    Returns:
    - A tuple containing the following element:
      - concat_r2s (float): The R2 score for the concatenated model predictions.

    """

    n_time, n_voxels = data.shape
    n_features = len(features)

    ind = get_cv_indices(n_time, n_folds=n_folds)

    # create arrays to store predictions
    concat_pred = np.zeros((n_time, n_voxels))

    # perform cross-validation by fold
    for ind_num in range(n_folds):
        # split data into training and testing sets
        train_ind = ind != ind_num
        test_ind = ind == ind_num
        train_data = data[train_ind]
        train_features = [F[train_ind] for F in features]
        test_data = data[test_ind]
        test_features = [F[test_ind] for F in features]

        # normalize data
        train_data = np.nan_to_num(zscore(train_data))
        test_data = np.nan_to_num(zscore(test_data))

        train_features = [np.nan_to_num(zscore(F)) for F in train_features]
        test_features = [np.nan_to_num(zscore(F)) for F in test_features]

        # Store predictions
        __,__, concat_pred[test_ind], __,__ = feat_ridge_CV(np.hstack(train_features), train_data, np.hstack(test_features), 
                                                            method=method)

        
    # Compute overall performance metrics
    data_zscored = zscore(data)

    concat_r2s = score_f(concat_pred, data_zscored)

    # return the results
    return (
        concat_r2s,
    )
