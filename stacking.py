from cvxopt import matrix, solvers

solvers.options["show_progress"] = False
from ridge_tools import *


def CV_ind(n, n_folds):
    # index for cross validation
    ind = np.zeros((n))
    n_items = int(np.floor(n / n_folds))  # number of items in one fold
    for i in range(0, n_folds - 1):
        ind[i * n_items : (i + 1) * n_items] = i
    ind[(n_folds - 1) * n_items :] = n_folds - 1
    return ind


def stacking_CV_fmri(data, features, method="cross_val_ridge", n_folds=4, score_f=R2):

    # INPUTS: data (ntime*nvoxels), features (list of ntime*ndim), method = what to use to train,
    #         n_folds = number of cross-val folds

    n_time = data.shape[0]
    n_voxels = data.shape[1]
    n_features = len(features)

    ind = CV_ind(n_time, n_folds=n_folds)

    # easier to store r2s in an array and access them programatically than to maintain a different
    # variable for each

    r2s = np.zeros((n_features, n_voxels))
    r2s_train_folds = np.zeros((n_folds, n_features, n_voxels))
    r2s_weighted = np.zeros((n_features, n_voxels))
    stacked_train_r2s_fold = np.zeros((n_folds, n_voxels))

    # store predictions in array
    stacked_pred = np.zeros((n_time, n_voxels))
    preds_test = np.zeros((n_features, n_time, n_voxels))
    weighted_pred = np.zeros((n_features, n_time, n_voxels))
    S_average = np.zeros((n_voxels, n_features))

    # DO BY FOLD
    for ind_num in range(n_folds):
        train_ind = ind != ind_num
        test_ind = ind == ind_num

        # split data
        train_data = data[train_ind]
        train_features = [F[train_ind] for F in features]

        test_data = data[test_ind]
        test_features = [F[test_ind] for F in features]

        # normalize data  by TRAIN/TEST
        train_data = np.nan_to_num(zscore(train_data))
        test_data = np.nan_to_num(zscore(test_data))

        train_features = [np.nan_to_num(zscore(F)) for F in train_features]
        test_features = [np.nan_to_num(zscore(F)) for F in test_features]

        err = dict()
        preds_train = dict()

        for FEATURE in range(n_features):
            if method == "simple_ridge":
                weights = ridge(train_features[FEATURE], train_data, 100)
            elif method == "cross_val_ridge":
                weights, __ = cross_val_ridge(
                    train_features[FEATURE],
                    train_data,
                    n_splits=4,
                    lambdas=np.array([10 ** i for i in range(-6, 10)]),
                    do_plot=False,
                )
            preds_train[FEATURE] = np.dot(train_features[FEATURE], weights)
            err[FEATURE] = train_data - preds_train[FEATURE]
            # predict the test data also before overwriting the weights:
            preds_test[FEATURE, test_ind] = np.dot(test_features[FEATURE], weights)

            # single feature space predictions, computed over a fold
            r2s_train_folds[ind_num, FEATURE, :] = score_f(
                preds_train[FEATURE], train_data
            )

        # calculate error matrix for stacking
        P = np.zeros((n_voxels, n_features, n_features))
        for i in range(n_features):
            for j in range(n_features):
                P[:, i, j] = np.mean(err[i] * err[j], 0)

        # PROGRAMATICALLY SET THIS FROM THE NUMBER OF FEATURES
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
            S[i, :] = np.array(solvers.qp(PP, q, G, h, A, b)["x"]).reshape(
                n_features,
            )

            # combine the predictions from the individual feature spaces for voxel i
            z = np.array(
                [preds_test[feature_j, test_ind, i] for feature_j in range(n_features)]
            )
            if i == 0:
                print(z.shape)  # to make sure
            # multiply the predictions by S[i,:]
            stacked_pred[test_ind, i] = np.dot(S[i, :], z)
            # combine the training predictions from the individual feature spaces for voxel i
            z = np.array(
                [preds_train[feature_j][:, i] for feature_j in range(n_features)]
            )
            stacked_pred_train[:, i] = np.dot(S[i, :], z)

        S_average += S

        stacked_train_r2s_fold[ind_num, :] = score_f(stacked_pred_train, train_data)

        for FEATURE in range(n_features):
            # weight the predictions according to S:
            # weighted single feature space predictions, computed over a fold
            weighted_pred[FEATURE, test_ind] = (
                preds_test[FEATURE, test_ind] * S[:, FEATURE]
            )

    # compute overall
    data_zscored = zscore(data)
    for FEATURE in range(n_features):
        r2s[FEATURE, :] = score_f(preds_test[FEATURE], data_zscored)
        r2s_weighted[FEATURE, :] = score_f(weighted_pred[FEATURE], data_zscored)

    stacked_r2s = score_f(stacked_pred, data_zscored)

    r2s_train = r2s_train_folds.mean(0)
    stacked_train = stacked_train_r2s_fold.mean(0)
    S_average = S_average / n_folds

    return (
        r2s,
        stacked_r2s,
        r2s_weighted,
        r2s_train,
        stacked_train,
        S_average,
    )
