import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ridge_tools import cross_val_ridge, fit_predict, R2
import sys
from stacking import stacking_CV_fmri
from concatenate import concatenate_CV_fmri
from scipy.stats import zscore, multivariate_normal #, wishart
from scipy.linalg import toeplitz
import time



def toeplitz_cov(n, scale=1):
    return toeplitz(np.exp(-(np.arange(n*1.0))**2/(n*scale)))

def feat_sample(n,ds,scale):
    nX = len(ds)
    d = sum(ds)
    Xtot = multivariate_normal.rvs(np.zeros(d),cov=toeplitz_cov(d,scale),size=n).T#reshape([n,d])
    Xs = []
    cnt=0
    for di in ds:
        Xs.append(Xtot[:,cnt:cnt+di])
        cnt = cnt +di
    return Xs

def data_sample(Xs,ds,scale,alpha,data_dim,noise=0):
    assert len(Xs) == len(alpha)
    #ws = []
    ts = []
    y = 0
    d = sum(ds)
    cnt = 0
    wtot = multivariate_normal.rvs(mean=np.zeros(d),cov=toeplitz_cov(d,scale),size=data_dim).T#reshape([d,data_dim])
    for iX, X in enumerate(Xs):
        w = wtot[cnt:cnt+ds[iX],:]
        cnt += ds[iX]
        t = zscore(X.dot(w))
        ts.append(t)
        y += alpha[iX]*t
    ns = noise*np.random.randn(y.shape[0], y.shape[1])
    y_orig = y
    y+= ns
    r_concat = R2(y_orig,y)
    var_X0 = R2(y_orig - ts[0]*alpha[0],y)
    return y, var_X0

def sample_all_at_once(n,ds,scale, alpha,data_dim,y_noise=0):
    Xs = feat_sample(n,ds,scale)
    y,var_X0  = data_sample(Xs,ds,scale,alpha,data_dim,y_noise)
    return Xs, y, var_X0  

def feat_sample(n,ds,scale,correl=0):
    Xs = []
    for di in ds:
        X = multivariate_normal.rvs(np.zeros(di),cov=toeplitz_cov(di,scale),size=n)#reshape([n,di])
        Xs.append(X)
    return Xs 

def data_sample(Xs,correl,ds,scale,alpha,data_dim,noise=0):
    assert len(Xs) == len(alpha)
    ts = []
    y = 0
    d = sum(ds)
    cnt = 0
    wtot = multivariate_normal.rvs(mean=np.zeros(d),cov=toeplitz_cov(d,scale),
                                          size=data_dim).T#reshape([d,data_dim])
    
    for iX, X in enumerate(Xs):
        w = wtot[cnt:cnt+ds[iX],:]
        cnt += ds[iX]
        t = zscore(X.dot(w))
        ts.append(t)
        y += alpha[iX]*t
    y = zscore(y)
    ns = noise*np.random.randn(y.shape[0], y.shape[1])
    y_orig = y
    y+= ns
    var_X = [R2(alpha[i]*ts[i],y) for i in range(len(Xs))]
    return y, var_X

def sample_all_at_once(n,ds,scale,correl, alpha,data_dim,y_noise=0):
    Xs = feat_sample(n,ds,scale,correl)
    y,var_X  = data_sample(Xs,correl,ds,scale,alpha,data_dim,y_noise)
    return Xs, y, var_X  

# Experiment Functions

import time

def synexp(runs,sim_type, samples_settings,ds_settings,y_dim,alpha_settings,correl = 0,scale = 1,
           y_noise_settings=0):

    Results = pd.DataFrame()
    
    start = time.time()
    for run in range(runs):
        print('iteration number {}'.format(run+1))
        if sim_type == 'Feat_Dim_ratio':  # Vary the dimensionality of X1 with respect to other feature spaces
            for ds in ds_settings:
                df = run_one_simulation(samples_settings,ds,scale,correl,alpha_settings,y_dim,
                                        y_noise_settings)
                df['Feat_Dim_ratio'] = ds[0]
                Results = pd.concat([Results,df],ignore_index=True)
        elif sim_type == 'Cond':          # Vary the weight of X1 with respect to other feature spaces
            for alpha in alpha_settings:
                df = run_one_simulation(samples_settings,ds_settings,scale,correl,alpha,y_dim,
                                        y_noise_settings)
                df['Cond'] = alpha[0]
                Results = pd.concat([Results,df],ignore_index=True)
        elif sim_type == 'Sample_Dim_ratio':   # Vary the number of samples
            for samples in samples_settings:
                df = run_one_simulation(samples,ds_settings,scale,correl,alpha_settings,y_dim,
                                        y_noise_settings)
                df['Sample_Dim_ratio'] = samples
                Results = pd.concat([Results,df],ignore_index=True)
        elif sim_type == 'noise':            # Vary the noise level
            for y_noise in y_noise_settings:
                df = run_one_simulation(samples_settings,ds_settings,scale,correl,alpha_settings,y_dim,
                                        y_noise)
                df['noise'] = y_noise
                Results = pd.concat([Results,df],ignore_index=True)
        elif sim_type == 'correl':            # Vary the feature space correlation level
            for correl_v in correl:
                df = run_one_simulation(samples_settings,ds_settings,scale,correl_v,alpha_settings,y_dim,
                                        y_noise_settings)
                df['correl'] = correl_v
                Results = pd.concat([Results,df],ignore_index=True)
                
        if run==0:
            time_int = (time.time() - start)
            print("first iteration time: {}, total {}".format(int(time_int), int(time_int*runs)))
                
        

    return Results



def run_one_simulation(samples,ds,scale,correl,alpha,y_dim,y_noise):
    Xs, y, var_X  = sample_all_at_once(samples,ds,scale,correl,alpha,y_dim,y_noise)
    print('data sampled')
    time_begin = time.time()
    y = zscore(y)

    concat_X = np.hstack(Xs)

    result = stacking_CV_fmri(y,Xs, method = 'cross_val_ridge',n_folds = 4)
    result2 = stacking_CV_fmri(y,Xs[1:], method = 'cross_val_ridge',n_folds = 4)

    print('time for stacking: {}'.format(time.time()-time_begin))

    df = pd.DataFrame()

    df['stacked'] = result[1]
    df['concat'] = concatenate_CV_fmri(y,Xs, method = 'cross_val_ridge',n_folds = 4)[0]
    df['max'] = np.max(result[0][0:2,:],axis=0)
    df['r2_X0'] = result[0][0]

    df['varpar_X0_concat'] = df['concat'] - concatenate_CV_fmri(y,Xs[1:], method = 'cross_val_ridge',n_folds = 4)[0]

    df['varpar_X0_stacked'] = df['stacked'] - result2[1]

    df['varpar_X0_real'] = var_X[0]

    df['weight_0'] = result[5][:,0]
    df['alpha_0'] = alpha[0]

    print('time for iteration: {}'.format(time.time()-time_begin))

    
    return df

