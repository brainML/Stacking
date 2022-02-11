# Functions to estimate cost for each lambda, by voxel:
from __future__ import division                                              

from numpy.linalg import inv, svd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, RidgeCV
import time 
from scipy.stats import zscore


def corr(X,Y,axis=0):
    #correlation coefficient
    return np.mean(zscore(X)*zscore(Y),axis)

def R2(Pred,Real):
    # coefficient of determination
    # R^2 = 1 -  residual sum of squares/total sum of squares 
    SSres = np.mean((Real-Pred)**2,0)
    SStot = np.var(Real,0)
    return np.nan_to_num(1-SSres/SStot)

def fit_predict(data, features, method='plain', n_folds=10):
    n, v = data.shape
    p = features.shape[1]
    corrs = np.zeros((n_folds, v))
    R2s = np.zeros((n_folds, v))
    ind = CV_ind(n, n_folds)
    preds_all = np.zeros_like(data)
    for i in range(n_folds):
        train_data = np.nan_to_num(zscore(data[ind != i]))
        train_features = np.nan_to_num(zscore(features[ind != i]))
        test_data = np.nan_to_num(zscore(data[ind == i]))
        test_features = np.nan_to_num(zscore(features[ind == i]))
        weights, __ = cross_val_ridge(train_features, train_data, method=method)
        preds = np.dot(test_features, weights)
        preds_all[ind == i] = preds
#         print("fold {}".format(i))
    corrs = corr(preds_all, data)
    R2s = R2(preds_all, data)
    return corrs, R2s

def CV_ind(n, n_folds):
    ind = np.zeros((n))
    n_items = int(np.floor(n/n_folds))
    for i in range(0,n_folds -1):
        ind[i*n_items:(i+1)*n_items] = i
    ind[(n_folds-1)*n_items:] = (n_folds-1)
    return ind

def R2r(Pred,Real):
    # square root of R^2
    R2rs = R2(Pred,Real)
    ind_neg = R2rs<0 # pick out negative ones
    R2rs = np.abs(R2rs)# use absolute value to calculate sqaure root
    R2rs = np.sqrt(R2rs)
    R2rs[ind_neg] *= - 1 # recover negative data
    return R2rs

def ridge(X,Y,lmbda):
    # weight of ridge regression
    return np.dot(inv(X.T.dot(X)+lmbda*np.eye(X.shape[1])),X.T.dot(Y))

def lasso(X,Y,lmbda):
    return soft_ths(ols(X,Y),X.shape[0]*lmbda)

def soft_ths(X,alpha):
    Y = np.zeros_like(X)
    Y[X > alpha] = (X - alpha)[X > alpha]
    Y[X < alpha] = (X + alpha)[X < alpha]

    return Y

# def soft_threshold(alpha, beta):
#     if beta > alpha:
#         return beta - alpha
#     elif beta < -alpha:
#         return beta + alpha
#     else:
#         return 0

def ols(X,Y):
    return np.dot(np.linalg.pinv(X.T.dot(X)),X.T.dot(Y))
    #return np.linalg.inv(X.T @ X) @ (X.T @ Y)

def ridge_by_lambda(X, Y, Xval, Yval, lambdas=np.array([0.1,1,10,100,1000])):
    # validation error of ridge regression under different lambdas
    error = np.zeros((lambdas.shape[0],Y.shape[1]))
    for idx,lmbda in enumerate(lambdas):
        weights = ridge(X,Y,lmbda)
        error[idx] = 1 - R2(np.dot(Xval,weights),Yval)
    return error

def lasso_by_lambda(X, Y, Xval, Yval, lambdas=np.array([0.1,1,10,100,1000])):
    # validation error of ridge regression under different lambdas
    error = np.zeros((lambdas.shape[0],Y.shape[1]))
    for idx,lmbda in enumerate(lambdas):
        weights = lasso(X,Y,lmbda)
        error[idx] = 1 - R2(np.dot(Xval,weights),Yval)
    return error

def ols_err(X, Y, Xval, Yval, lambdas=np.array([0.1,1,10,100,1000])):
    error = np.zeros(Y.shape[1])
    weights = ols(X,Y)
    error= 1 - R2(np.dot(Xval,weights),Yval)
    return error


def ridge_sk(X,Y,lmbda):
    rd = Ridge(alpha = lmbda)
    rd.fit(X,Y)
    return rd.coef_.T

def ridgeCV_sk(X,Y,lmbdas):
    rd = RidgeCV(alphas = lmbdas,solver = 'svd')
    rd.fit(X,Y)
    return rd.coef_.T

def ridge_by_lambda_sk(X, Y, Xval, Yval, lambdas=np.array([0.1,1,10,100,1000])):
    error = np.zeros((lambdas.shape[0],Y.shape[1]))
    for idx,lmbda in enumerate(lambdas):
        weights = ridge_sk(X,Y,lmbda)
        error[idx] = 1 -  R2(np.dot(Xval,weights),Yval)
    return error

def ridge_svd(X,Y,lmbda):
    U, s, Vt = svd(X, full_matrices=False)
    d = s / (s** 2 + lmbda)
    return np.dot(Vt,np.diag(d).dot(U.T.dot(Y)))

def ridge_by_lambda_svd(X, Y, Xval, Yval, lambdas=np.array([0.1,1,10,100,1000])):
    error = np.zeros((lambdas.shape[0],Y.shape[1]))
    U, s, Vt = svd(X, full_matrices=False)
    for idx,lmbda in enumerate(lambdas):
        d = s / (s** 2 + lmbda)
        weights = np.dot(Vt,np.diag(d).dot(U.T.dot(Y)))
        error[idx] = 1 - R2(np.dot(Xval,weights),Yval)
    return error


def kernel_ridge(X,Y,lmbda):
    return np.dot(X.T.dot(inv(X.dot(X.T)+lmbda*np.eye(X.shape[0]))),Y)

def kernel_ridge_by_lambda(X, Y, Xval, Yval, lambdas=np.array([0.1,1,10,100,1000])):
    error = np.zeros((lambdas.shape[0],Y.shape[1]))
    for idx,lmbda in enumerate(lambdas):
        weights = kernel_ridge(X,Y,lmbda)
        error[idx] = 1 - R2(np.dot(Xval,weights),Yval)
    return error


def kernel_ridge_svd(X,Y,lmbda):
    U, s, Vt = svd(X.T, full_matrices=False)
    d = s / (s** 2 + lmbda)
    return np.dot(np.dot(U,np.diag(d).dot(Vt)),Y)

def kernel_ridge_by_lambda_svd(X, Y, Xval, Yval, lambdas=np.array([0.1,1,10,100,1000])):
    error = np.zeros((lambdas.shape[0],Y.shape[1]))
    U, s, Vt = svd(X.T, full_matrices=False)
    for idx,lmbda in enumerate(lambdas):
        d = s / (s** 2 + lmbda)
        weights = np.dot(np.dot(U,np.diag(d).dot(Vt)),Y)
        error[idx] = 1 - R2(np.dot(Xval,weights),Yval)
    return error



def cross_val_ridge(train_features,train_data, n_splits = 10, 
                    lambdas = np.array([10**i for i in range(-6,10)]),
                    method = 'plain',
                    do_plot = False):
# cross validation for ridge regression

    ridge_1 = dict(plain = ridge_by_lambda,
                   svd = ridge_by_lambda_svd,
                   kernel_ridge = kernel_ridge_by_lambda,
                   kernel_ridge_svd = kernel_ridge_by_lambda_svd,
                   ridge_sk = ridge_by_lambda_sk)[method] #loss of the regressor
    ridge_2 = dict(plain = ridge,
                   svd = ridge_svd,
                   kernel_ridge = kernel_ridge,
                   kernel_ridge_svd = kernel_ridge_svd,
                   ridge_sk = ridge_sk)[method] # solver for the weights
    
    n_voxels = train_data.shape[1] # get number of voxels from data
    nL = lambdas.shape[0] # get number of hyperparameter (lambdas) from setting 
    r_cv = np.zeros((nL, train_data.shape[1])) # loss matrix

    kf = KFold(n_splits=n_splits) # set up dataset for cross validation
    start_t = time.time() # record start time 
    for icv, (trn, val) in enumerate(kf.split(train_data)):
        # print('ntrain = {}'.format(train_features[trn].shape[0]))
        cost = ridge_1(train_features[trn],train_data[trn],
                               train_features[val],train_data[val], 
                               lambdas=lambdas) # loss of regressor 1
        if do_plot: 
            import matplotlib.pyplot as plt
            plt.figure()
            plt.imshow(cost,aspect = 'auto')
        r_cv += cost
#         if icv%3 ==0:
#             print(icv)
#         print('average iteration length {}'.format((time.time()-start_t)/(icv+1))) # time used
    if do_plot: # show loss
        plt.figure()
        plt.imshow(r_cv,aspect='auto',cmap = 'RdBu_r');

    argmin_lambda = np.argmin(r_cv,axis = 0) # pick the best lambda
    weights = np.zeros((train_features.shape[1],train_data.shape[1])) # initialize the weight
    for idx_lambda in range(lambdas.shape[0]): # this is much faster than iterating over voxels!
        idx_vox = argmin_lambda == idx_lambda
        weights[:,idx_vox] = ridge_2(train_features, train_data[:,idx_vox],lambdas[idx_lambda])
    if do_plot: # show the weights
        plt.figure()
        plt.imshow(weights,aspect='auto',cmap = 'RdBu_r',vmin = -0.5,vmax = 0.5);

    return weights, np.array([lambdas[i] for i in argmin_lambda])



def cross_val_lasso(train_features,train_data, n_splits = 10, 
                    lambdas = np.array([10**i for i in range(-6,10)]),
                    method = 'plain',
                    do_plot = False):
# cross validation for ridge regression

    # ridge_1 = dict(plain = ridge_by_lambda,
    #                svd = ridge_by_lambda_svd,
    #                kernel_ridge = kernel_ridge_by_lambda,
    #                kernel_ridge_svd = kernel_ridge_by_lambda_svd,
    #                ridge_sk = ridge_by_lambda_sk)[method] #loss of the regressor
    # ridge_2 = dict(plain = ridge,
    #                svd = ridge_svd,
    #                kernel_ridge = kernel_ridge,
    #                kernel_ridge_svd = kernel_ridge_svd,
    #                ridge_sk = ridge_sk)[method] # solver for the weights
    
    n_voxels = train_data.shape[1] # get number of voxels from data
    nL = lambdas.shape[0] # get number of hyperparameter (lambdas) from setting 
    r_cv = np.zeros((nL, train_data.shape[1])) # loss matrix

    kf = KFold(n_splits=n_splits) # set up dataset for cross validation
    start_t = time.time() # record start time 
    for icv, (trn, val) in enumerate(kf.split(train_data)):
        print('ntrain = {}'.format(train_features[trn].shape[0]))
        cost = lasso_by_lambda(train_features[trn],train_data[trn],
                               train_features[val],train_data[val], 
                               lambdas=lambdas) # loss of regressor 1
        if do_plot: 
            import matplotlib.pyplot as plt
            plt.figure()
            plt.imshow(cost,aspect = 'auto')
        r_cv += cost
        if icv%3 ==0:
            print(icv)
        print('average iteration length {}'.format((time.time()-start_t)/(icv+1))) # time used
    if do_plot: # show loss
        plt.figure()
        plt.imshow(r_cv,aspect='auto',cmap = 'RdBu_r');

    argmin_lambda = np.argmin(r_cv,axis = 0) # pick the best lambda
    weights = np.zeros((train_features.shape[1],train_data.shape[1])) # initialize the weight
    for idx_lambda in range(lambdas.shape[0]): # this is much faster than iterating over voxels!
        idx_vox = argmin_lambda == idx_lambda
        weights[:,idx_vox] = lasso(train_features, train_data[:,idx_vox],lambdas[idx_lambda])
    if do_plot: # show the weights
        plt.figure()
        plt.imshow(weights,aspect='auto',cmap = 'RdBu_r',vmin = -0.5,vmax = 0.5);

    return weights, np.array([lambdas[i] for i in argmin_lambda])


def cross_val_ols(train_features,train_data, n_splits = 10, 
                    lambdas = np.array([10**i for i in range(-6,10)]),
                    method = 'plain',
                    do_plot = False):
# cross validation for ridge regression

    # ridge_1 = dict(plain = ridge_by_lambda,
    #                svd = ridge_by_lambda_svd,
    #                kernel_ridge = kernel_ridge_by_lambda,
    #                kernel_ridge_svd = kernel_ridge_by_lambda_svd,
    #                ridge_sk = ridge_by_lambda_sk)[method] #loss of the regressor
    # ridge_2 = dict(plain = ridge,
    #                svd = ridge_svd,
    #                kernel_ridge = kernel_ridge,
    #                kernel_ridge_svd = kernel_ridge_svd,
    #                ridge_sk = ridge_sk)[method] # solver for the weights
    
    n_voxels = train_data.shape[1] # get number of voxels from data
    nL = lambdas.shape[0] # get number of hyperparameter (lambdas) from setting 
    r_cv = np.zeros((nL, train_data.shape[1])) # loss matrix

    kf = KFold(n_splits=n_splits) # set up dataset for cross validation
    start_t = time.time() # record start time 
    for icv, (trn, val) in enumerate(kf.split(train_data)):
        print('ntrain = {}'.format(train_features[trn].shape[0]))
        cost = ols_err(train_features[trn],train_data[trn],
                               train_features[val],train_data[val], 
                               lambdas=lambdas) # loss of regressor 1
        if do_plot: 
            import matplotlib.pyplot as plt
            plt.figure()
            plt.imshow(cost,aspect = 'auto')
        r_cv += cost
        if icv%3 ==0:
            print(icv)
        print('average iteration length {}'.format((time.time()-start_t)/(icv+1))) # time used
    if do_plot: # show loss
        plt.figure()
        plt.imshow(r_cv,aspect='auto',cmap = 'RdBu_r');

    argmin_lambda = np.argmin(r_cv,axis = 0) # pick the best lambda
    weights = np.zeros((train_features.shape[1],train_data.shape[1])) # initialize the weight
    for idx_lambda in range(lambdas.shape[0]): # this is much faster than iterating over voxels!
        idx_vox = argmin_lambda == idx_lambda
        weights[:,idx_vox] = ols(train_features, train_data[:,idx_vox])
    if do_plot: # show the weights
        plt.figure()
        plt.imshow(weights,aspect='auto',cmap = 'RdBu_r',vmin = -0.5,vmax = 0.5);

    return weights, np.array([lambdas[i] for i in argmin_lambda])





def GCV_ridge(train_features,train_data,lambdas = np.array([10**i for i in range(-6,10)])):
    
    n_lambdas = lambdas.shape[0]
    n_voxels = train_data.shape[1]
    n_time = train_data.shape[0]
    n_p = train_features.shape[1]

    CVerr = np.zeros((n_lambdas, n_voxels))

    # % If we do an eigendecomp first we can quickly compute the inverse for many different values
    # % of lambda. SVD uses X = UDV' form.
    # % First compute K0 = (X'X + lambda*I) where lambda = 0.
    #K0 = np.dot(train_features,train_features.T)
    print('Running svd',)
    start_time = time.time()
    [U,D,Vt] = svd(train_features,full_matrices=False)
    V = Vt.T
    print(U.shape,D.shape,Vt.shape)
    print('svd time: {}'.format(time.time() - start_time))

    for i,regularizationParam in enumerate(lambdas):
        regularizationParam = lambdas[i]
        print('CVLoop: Testing regularization param: {}'.format(regularizationParam))

        #Now we can obtain Kinv for any lambda doing Kinv = V * (D + lambda*I)^-1 U'
        dlambda = D**2 + np.eye(n_p)*regularizationParam
        dlambdaInv = np.diag(D / np.diag(dlambda))
        KlambdaInv = V.dot(dlambdaInv).dot(U.T)
        
        # Compute S matrix of Hastie Trick  H = X(XT X + lambdaI)-1XT
        S = np.dot(U, np.diag(D * np.diag(dlambdaInv))).dot(U.T)
        denum = 1-np.trace(S)/n_time
        
        # Solve for weight matrix so we can compute residual
        weightMatrix = KlambdaInv.dot(train_data);


#         Snorm = np.tile(1 - np.diag(S) , (n_voxels, 1)).T
        YdiffMat = (train_data - (train_features.dot(weightMatrix)));
        YdiffMat = YdiffMat / denum;
        CVerr[i,:] = (1/n_time)*np.sum(YdiffMat * YdiffMat,0);


    # try using min of avg err
    minerrIndex = np.argmin(CVerr,axis = 0);
    r=np.zeros((n_voxels));

    for nPar,regularizationParam in enumerate(lambdas):
        ind = np.where(minerrIndex==nPar)[0];
        if len(ind)>0:
            r[ind] = regularizationParam;
            print('{}% of outputs with regularization param: {}'.format(int(len(ind)/n_voxels*100),
                                                                        regularizationParam))
            # got good param, now obtain weights
            dlambda = D**2 + np.eye(n_p)*regularizationParam
            dlambdaInv = np.diag(D / np.diag(dlambda))
            KlambdaInv = V.dot(dlambdaInv).dot(U.T)

            weightMatrix[:,ind] = KlambdaInv.dot(train_data[:,ind]);


    return weightMatrix, r

# def cross_val_ridge_tikhonov(train_features,train_data, n_splits = 10, 
#                     lambdas = np.array([10**i for i in range(-6,10)]),
#                     method = 'plain',
#                     do_plot = False):
    
#     ridge_1 = dict(plain = ridge_by_lambda,
#                    svd = ridge_by_lambda_svd,
#                    kernel_ridge = kernel_ridge_by_lambda,
#                    kernel_ridge_svd = kernel_ridge_by_lambda_svd,
#                    ridge_sk = ridge_by_lambda_sk)[method]
#     ridge_2 = dict(plain = ridge,
#                    svd = ridge_svd,
#                    kernel_ridge = kernel_ridge,
#                    kernel_ridge_svd = kernel_ridge_svd,
#                    ridge_sk = ridge_sk)[method]
    
#     n_voxels = train_data.shape[1]
#     r_cv = lambdas.shape.append(train_data.shape[1])
#     r_cv = np.zeros(r_cv)

#     kf = KFold(n_splits=n_splits)
#     start_t = time.time()
#     for icv, (trn, val) in enumerate(kf.split(train_data)):
#         print('ntrain = {}'.format(train_features[trn].shape[0]))
#         cost = ridge_1(train_features[trn],train_data[trn],
#                                train_features[val],train_data[val], 
#                                lambdas=lambdas)
#         if do_plot:
#             import matplotlib.pyplot as plt
#             plt.figure()
#             plt.imshow(cost,aspect = 'auto')
#         r_cv += cost
#         if icv%3 ==0:
#             print(icv)
#         print('average iteration length {}'.format((time.time()-start_t)/(icv+1)))
#     if do_plot:
#         plt.figure()
#         plt.imshow(r_cv,aspect='auto',cmap = 'RdBu_r');

#     argmin_lambda = np.argmin(r_cv,axis = 0)
#     weights = np.zeros((train_features.shape[1],train_data.shape[1]))
#     for idx_lambda in range(lambdas.shape[0]): # this is much faster than iterating over voxels!
#         idx_vox = argmin_lambda == idx_lambda
#         weights[:,idx_vox] = ridge_2(train_features, train_data[:,idx_vox],lambdas[idx_lambda])
#     if do_plot:
#         plt.figure()
#         plt.imshow(weights,aspect='auto',cmap = 'RdBu_r',vmin = -0.5,vmax = 0.5);

#     return weights, np.array([lambdas[i] for i in argmin_lambda])

    
