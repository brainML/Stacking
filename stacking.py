from cvxopt import matrix, solvers
import numpy as np
from scipy.stats import zscore
from ridge_tools import cross_val_ridge, R2, ridge

# Set option to not show progress in CVXOPT solver
solvers.options["show_progress"] = False


def get_cv_indices(n_samples, n_folds):
    """Generate cross-validation indices.

    Args:
        n_samples (int): Number of samples to generate indices for.
        n_folds (int): Number of folds to use in cross-validation.

    Returns:
        numpy.ndarray: Array of cross-validation indices with shape (n_samples,).
    """
    cv_indices = np.zeros((n_samples))
    n_items = int(np.floor(n_samples / n_folds))  # number of items in one fold
    for i in range(0, n_folds - 1):
        cv_indices[i * n_items : (i + 1) * n_items] = i
    cv_indices[(n_folds - 1) * n_items :] = n_folds - 1
    return cv_indices


def feat_ridge_CV(
    train_features,
    train_targets,
    test_features,
    method="cross_val_ridge",
    n_folds=5,
    score_function=R2,
):
    """Train a ridge regression model with cross-validation and predict on test_features.

    Args:
        train_features (numpy.ndarray): Array of shape (n_samples, n_features) containing the training features.
        train_targets (numpy.ndarray): Array of shape (n_samples, n_targets) containing the training targets.
        test_features (numpy.ndarray): Array of shape (n_test_samples, n_features) containing the test features.
        method (str): Method to use for ridge regression. Options are "simple_ridge" and "cross_val_ridge".
            Defaults to "cross_val_ridge".
        n_folds (int): Number of folds to use in cross-validation. Defaults to 5.
        score_function (callable): Scoring function to use for cross-validation. Defaults to R2.

    Returns:
        tuple: Tuple containing:
            - preds_train (numpy.ndarray): Array of shape (n_samples, n_targets) containing the training set predictions.
            - err (numpy.ndarray): Array of shape (n_samples, n_targets) containing the training set errors.
            - preds_test (numpy.ndarray): Array of shape (n_test_samples, n_targets) containing the test set predictions.
            - r2s_train_fold (numpy.ndarray): Array of shape (n_folds,) containing the cross-validation scores.
            - var_train_fold (numpy.ndarray): Array of shape (n_targets,) containing the variances of the training set predictions.
    """

    if np.all(train_features == 0):
        # If there are no predictors, return zero weights and zero predictions
        weights = np.zeros((train_features.shape[1], train_targets.shape[1]))
        train_preds = np.zeros_like(train_targets)
    else:
        # Use cross-validation to train the model
        cv_indices = get_cv_indices(train_targets.shape[0], n_folds=n_folds)
        train_preds = np.zeros_like(train_targets)

        for i_cv in range(n_folds):
            train_targets_cv = np.nan_to_num(zscore(train_targets[cv_indices != i_cv]))
            train_features_cv = np.nan_to_num(
                zscore(train_features[cv_indices != i_cv])
            )
            test_features_cv = np.nan_to_num(zscore(train_features[cv_indices == i_cv]))

            if method == "simple_ridge":
                # Use a fixed regularization parameter to train the model
                weights = ridge(train_features, train_targets, 100)
            elif method == "cross_val_ridge":
                # Use cross-validation to select the best regularization parameter
                lambdas = np.array([10**i for i in range(-6, 10)])
                if train_features.shape[1] > train_features.shape[0]:
                    weights, __ = cross_val_ridge(
                        train_features_cv,
                        train_targets_cv,
                        n_splits=5,
                        lambdas=lambdas,
                        do_plot=False,
                        method="plain",
                    )
                else:
                    weights, __ = cross_val_ridge(
                        train_features_cv,
                        train_targets_cv,
                        n_splits=5,
                        lambdas=lambdas,
                        do_plot=False,
                        method="plain",
                    )

            # Make predictions on the current fold of the data
            train_preds[cv_indices == i_cv] = test_features_cv.dot(weights)

    # Calculate prediction error on the training set
    train_err = train_targets - train_preds

    # Retrain the model on all of the training data
    lambdas = np.array([10**i for i in range(-6, 10)])
    weights, __ = cross_val_ridge(
        train_features,
        train_targets,
        n_splits=5,
        lambdas=lambdas,
        do_plot=False,
        method="plain",
    )

    # Make predictions on the test set using the retrained model
    test_preds = np.dot(test_features, weights)

    # Calculate the score on the training set
    train_scores = score_function(train_preds, train_targets)
    train_variances = np.var(train_preds, axis=0)

    return train_preds, train_err, test_preds, train_scores, train_variances


import numpy as np
from cvxopt import matrix, solvers


def stacking_fmri(
    train_data,
    test_data,
    train_features,
    test_features,
    method="cross_val_ridge",
    score_f=R2,
):
    """
    Stacks predictions from different feature spaces and uses them to make final predictions.

    Args:
        train_data (ndarray): Training data of shape (n_time_train, n_voxels)
        test_data (ndarray): Testing data of shape (n_time_test, n_voxels)
        train_features (list): List of training feature spaces, each of shape (n_time_train, n_dims)
        test_features (list): List of testing feature spaces, each of shape (n_time_test, n_dims)
        method (str): Name of the method used for training. Default is 'cross_val_ridge'.
        score_f (callable): Scikit-learn scoring function to use for evaluation. Default is mean_squared_error.

    Returns:
        Tuple of ndarrays:
            - r2s: Array of shape (n_features, n_voxels) containing unweighted R2 scores for each feature space and voxel
            - stacked_r2s: Array of shape (n_voxels,) containing R2 scores for the stacked predictions of each voxel
            - r2s_weighted: Array of shape (n_features, n_voxels) containing R2 scores for each feature space weighted by stacking weights
            - r2s_train: Array of shape (n_features, n_voxels) containing R2 scores for each feature space and voxel in the training set
            - stacked_train_r2s: Array of shape (n_voxels,) containing R2 scores for the stacked predictions of each voxel in the training set
            - S: Array of shape (n_voxels, n_features) containing the stacking weights for each voxel
    """

    # Number of time points in the test set
    n_time_test = test_data.shape[0]

    # Check that the number of voxels is the same in the training and test sets
    assert train_data.shape[1] == test_data.shape[1]
    n_voxels = train_data.shape[1]

    # Check that the number of feature spaces is the same in the training and test sets
    assert len(train_features) == len(test_features)
    n_features = len(train_features)

    # Array to store R2 scores for each feature space and voxel
    r2s = np.zeros((n_features, n_voxels))
    # Array to store R2 scores for each feature space and voxel in the training set
    r2s_train = np.zeros((n_features, n_voxels))
    # Array to store variance explained by the model for each feature space and voxel in the training set
    var_train = np.zeros((n_features, n_voxels))
    # Array to store R2 scores for each feature space weighted by stacking weights
    r2s_weighted = np.zeros((n_features, n_voxels))

    # Array to store stacked predictions for each voxel
    stacked_pred = np.zeros((n_time_test, n_voxels))
    # Dictionary to store predictions for each feature space and voxel in the training set
    preds_train = {}
    # Dictionary to store predictions for each feature space and voxel in the test set
    preds_test = np.zeros((n_features, n_time_test, n_voxels))
    # Array to store weighted predictions for each feature space and voxel in the test set
    weighted_pred = np.zeros((n_features, n_time_test, n_voxels))

    # normalize data by TRAIN/TEST
    train_data = np.nan_to_num(zscore(train_data))
    test_data = np.nan_to_num(zscore(test_data))

    train_features = [np.nan_to_num(zscore(F)) for F in train_features]
    test_features = [np.nan_to_num(zscore(F)) for F in test_features]

    # initialize an error dictionary to store errors for each feature
    err = dict()
    preds_train = dict()

    # iterate over each feature and train a model using feature ridge regression
    for FEATURE in range(n_features):
        (
            preds_train[FEATURE],
            error,
            preds_test[FEATURE, :, :],
            r2s_train[FEATURE, :],
            var_train[FEATURE, :],
        ) = feat_ridge_CV(
            train_features[FEATURE], train_data, test_features[FEATURE], method=method
        )
        err[FEATURE] = error

    # calculate error matrix for stacking
    P = np.zeros((n_voxels, n_features, n_features))
    for i in range(n_features):
        for j in range(n_features):
            P[:, i, j] = np.mean(err[i] * err[j], 0)

    # solve the quadratic programming problem to obtain the weights for stacking
    q = matrix(np.zeros((n_features)))
    G = matrix(-np.eye(n_features, n_features))
    h = matrix(np.zeros(n_features))
    A = matrix(np.ones((1, n_features)))
    b = matrix(np.ones(1))

    S = np.zeros((n_voxels, n_features))
    stacked_pred_train = np.zeros_like(train_data)

    for i in range(0, n_voxels):
        PP = matrix(P[i])
        # solve for stacking weights for every voxel
        S[i, :] = np.array(solvers.qp(PP, q, G, h, A, b)["x"]).reshape(n_features)

        # combine the predictions from the individual feature spaces for voxel i
        z_test = np.array(
            [preds_test[feature_j, :, i] for feature_j in range(n_features)]
        )
        z_train = np.array(
            [preds_train[feature_j][:, i] for feature_j in range(n_features)]
        )
        # multiply the predictions by S[i,:]
        stacked_pred[:, i] = np.dot(S[i, :], z_test)
        # combine the training predictions from the individual feature spaces for voxel i
        stacked_pred_train[:, i] = np.dot(S[i, :], z_train)

    # compute the R2 score for the stacked predictions on the training data
    stacked_train_r2s = score_f(stacked_pred_train, train_data)

    # compute the R2 scores for each individual feature and the weighted feature predictions
    for FEATURE in range(n_features):
        # weight the predictions according to S:
        # weighted single feature space predictions, computed over a fold
        weighted_pred[FEATURE, :] = preds_test[FEATURE, :] * S[:, FEATURE]

    for FEATURE in range(n_features):
        r2s[FEATURE, :] = score_f(preds_test[FEATURE], test_data)
        r2s_weighted[FEATURE, :] = score_f(weighted_pred[FEATURE], test_data)

    # compute the R2 score for the stacked predictions on the test data
    stacked_r2s = score_f(stacked_pred, test_data)

    # return the results
    return (
        r2s,
        stacked_r2s,
        r2s_weighted,
        r2s_train,
        stacked_train_r2s,
        S,
    )


def stacking_CV_fmri(data, features, method="cross_val_ridge", n_folds=5, score_f=R2):
    """
    A function that performs cross-validated feature stacking to predict fMRI
    signal from a set of predictors.

    Args:
    - data (ndarray): A matrix of fMRI signal data with dimensions n_time x n_voxels.
    - features (list): A list of length n_features containing arrays of predictors
      with dimensions n_time x n_dim.
    - method (str): A string indicating the method to use to train the model. Default is "cross_val_ridge".
    - n_folds (int): An integer indicating the number of cross-validation folds to use. Default is 5.
    - score_f (function): A function to use for scoring the model. Default is R2.

    Returns:
    - A tuple containing the following elements:
      - r2s (ndarray): An array of shape (n_features, n_voxels) containing the R2 scores
        for each feature and voxel.
      - r2s_weighted (ndarray): An array of shape (n_features, n_voxels) containing the R2 scores
        for each feature and voxel, weighted by stacking weights.
      - stacked_r2s (float): The R2 score for the stacked predictions.
      - r2s_train (ndarray): An array of shape (n_features, n_voxels) containing the R2 scores
        for each feature and voxel for the training set.
      - stacked_train (float): The R2 score for the stacked predictions for the training set.
      - S_average (ndarray): An array of shape (n_features, n_voxels) containing the stacking weights
        for each feature and voxel.

    """

    n_time, n_voxels = data.shape
    n_features = len(features)

    ind = get_cv_indices(n_time, n_folds=n_folds)

    # create arrays to store results
    r2s = np.zeros((n_features, n_voxels))
    r2s_train_folds = np.zeros((n_folds, n_features, n_voxels))
    var_train_folds = np.zeros((n_folds, n_features, n_voxels))
    r2s_weighted = np.zeros((n_features, n_voxels))
    stacked_train_r2s_fold = np.zeros((n_folds, n_voxels))
    stacked_pred = np.zeros((n_time, n_voxels))
    preds_test = np.zeros((n_features, n_time, n_voxels))
    weighted_pred = np.zeros((n_features, n_time, n_voxels))
    S_average = np.zeros((n_voxels, n_features))

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

        # Store prediction errors and training predictions for each feature
        err = dict()
        preds_train = dict()
        for FEATURE in range(n_features):
            (
                preds_train[FEATURE],
                error,
                preds_test[FEATURE, test_ind],
                r2s_train_folds[ind_num, FEATURE, :],
                var_train_folds[ind_num, FEATURE, :],
            ) = feat_ridge_CV(
                train_features[FEATURE],
                train_data,
                test_features[FEATURE],
                method=method,
            )
            err[FEATURE] = error

        # calculate error matrix for stacking
        P = np.zeros((n_voxels, n_features, n_features))
        for i in range(n_features):
            for j in range(n_features):
                P[:, i, j] = np.mean(err[i] * err[j], axis=0)

        # Set optimization parameters for computing stacking weights
        q = matrix(np.zeros((n_features)))
        G = matrix(-np.eye(n_features, n_features))
        h = matrix(np.zeros(n_features))
        A = matrix(np.ones((1, n_features)))
        b = matrix(np.ones(1))

        S = np.zeros((n_voxels, n_features))
        stacked_pred_train = np.zeros_like(train_data)

        # Compute stacking weights and combined predictions for each voxel
        for i in range(n_voxels):
            PP = matrix(P[i])
            # solve for stacking weights for every voxel
            S[i, :] = np.array(solvers.qp(PP, q, G, h, A, b)["x"]).reshape(
                n_features,
            )
            # combine the predictions from the individual feature spaces for voxel i
            z = np.array(
                [preds_test[feature_j, test_ind, i] for feature_j in range(n_features)]
            )
            # multiply the predictions by S[i,:]
            stacked_pred[test_ind, i] = np.dot(S[i, :], z)
            # combine the training predictions from the individual feature spaces for voxel i
            z = np.array(
                [preds_train[feature_j][:, i] for feature_j in range(n_features)]
            )
            stacked_pred_train[:, i] = np.dot(S[i, :], z)

        S_average += S
        stacked_train_r2s_fold[ind_num, :] = score_f(stacked_pred_train, train_data)

        # Compute weighted single feature space predictions, computed over a fold
        for FEATURE in range(n_features):
            weighted_pred[FEATURE, test_ind] = (
                preds_test[FEATURE, test_ind] * S[:, FEATURE]
            )

    # Compute overall performance metrics
    data_zscored = zscore(data)
    for FEATURE in range(n_features):
        r2s[FEATURE, :] = score_f(preds_test[FEATURE], data_zscored)
        r2s_weighted[FEATURE, :] = score_f(weighted_pred[FEATURE], data_zscored)

    stacked_r2s = score_f(stacked_pred, data_zscored)

    r2s_train = r2s_train_folds.mean(0)
    stacked_train = stacked_train_r2s_fold.mean(0)
    S_average = S_average / n_folds

    # return the results
    return (
        r2s,
        stacked_r2s,
        r2s_weighted,
        r2s_train,
        stacked_train,
        S_average,
    )
