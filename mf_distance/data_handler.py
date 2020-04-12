# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 11:43:35 2019

@author: 119275
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def uni_distributed(r_seed, m, n, mv_config, drift_config=None, mr_config=None, output_files=None):
    
    np.random.seed(r_seed)
    
    print('='*50)
    print('|' + 'uni_ind_distributed'.center(48) + '|')
    print('|' + ('r_seed:'+ str(r_seed) + ', m:' + str(m) + ', n:' + str(n)).center(48) + '|')
    print('|' + str(mv_config).center(48) + '|')
    print('='*50)

    data = np.random.uniform(size = [m, n])
    
    if mr_config is None:
        data_nan, data = fillin_nan_MCAR(r_seed, mv_config, data)
        if drift_config is not None:
            data_nan[:, :2] = data_nan[:, :2] + drift_config
            data[:, :2] = data[:, :2] + drift_config
            
    else:
        data_nan, data = fillin_nan_MNAR(r_seed, mv_config, data, mr_config)
        if drift_config is not None:
            data_nan[:, :2] = data_nan[:, :2] + drift_config
            data[:, :2] = data[:, :2] + drift_config
            
    if output_files is not None:
        np.savetxt(output_files[0], data_nan, delimiter=',', fmt='%.5f')
        np.savetxt(output_files[1], data, delimiter=',', fmt='%.5f')
    
    return data_nan, data

def gau_distributed(r_seed, m, n, mv_config, mu_sigma=None, drift_config=None, mr_config=None, output_files=None):
    
    np.random.seed(r_seed)
    
    print('='*50)
    print('|' + 'gau_ind_distributed'.center(48) + '|')
    print('|' + ('r_seed:'+ str(r_seed) + ', m:' + str(m) + ', n:' + str(n)).center(48) + '|')
    print('|' + str(mv_config).center(48) + '|')
    print('='*50)
    
    if mu_sigma is None:
        mu = np.random.rand(n)*10
        #sigma = np.random.rand(n)*5
        sigma = np.ones(n)
        mu_sigma = [mu, sigma]
    else:
        mu = mu_sigma[0]
        sigma = mu_sigma[1]
        
    cov = np.zeros([n, n])
    np.fill_diagonal(cov, sigma)
    
    data = np.random.multivariate_normal(mu, cov, m)
    data_nan, data = fillin_nan_MCAR(r_seed, mv_config, data)
    
    if mr_config is None:
        if drift_config is not None:
            if drift_config[0] == 'mean':
                data_nan[:, :2] = data_nan[:, :2] + drift_config[1]
                data[:, :2] = data[:, :2] + drift_config[1]
            elif drift_config[0] == 'cov':
                cov[0, 1] = drift_config[1]
                cov[1, 0] = drift_config[1]
                data = np.random.multivariate_normal(mu, cov, m)
                data_nan, data = fillin_nan_MCAR(r_seed, mv_config, data)
    else:
        if drift_config is not None:
            # ('mean', delta)
            if drift_config[0] == 'mean':
                data_nan[:, :2] = data_nan[:, :2] + drift_config[1]
                data[:, :2] = data[:, :2] + drift_config[1]
            # ('cov', delta)
            elif drift_config[0] == 'cov':
                cov[0, 1] = drift_config[1]
                cov[1, 0] = drift_config[1]
                data = np.random.multivariate_normal(mu, cov, m)
                data_nan, data = fillin_nan_MCAR(r_seed, mv_config, data)
        
    if output_files is not None:
        np.savetxt(output_files[0], data_nan, delimiter=',', fmt='%.5f')
        np.savetxt(output_files[1], data, delimiter=',', fmt='%.5f')
        
    return data_nan, data, mu_sigma

def poi_distributed(r_seed, m, n, mv_config, lam=None, rho=None, drift_config=None):
    
    np.random.seed(r_seed)
    
    print('='*50)
    print('|' + 'poi_ind_distributed'.center(48) + '|')
    print('|' + ('r_seed:'+ str(r_seed) + ', m:' + str(m) + ', n:' + str(n)).center(48) + '|')
    print('|' + str(mv_config).center(48) + '|')
    print('='*50)
    
    if lam is None:
        lam = np.random.randint(low=5, high=50, size=(n))
    if drift_config is not None:
        lam_drift = np.array(lam)
        lam_drift[:2] = lam[:2] + drift_config
    else:
        lam_drift = lam
    data = np.random.poisson(lam_drift, size=(m, n))*1.0
    if rho is not None:
        data[:, 0] = np.random.poisson(lam[0]*(1-rho), m)*1.0
        data[:, 1] = np.random.poisson(lam[0]*(1-rho), m)*1.0
        data_rho = np.random.poisson(lam[0]*rho, m)*1.0
        data[:, 0] = data[:, 0] + data_rho
        data[:, 1] = data[:, 1] + data_rho
    
    data_nan, data = fillin_nan_MCAR(r_seed, mv_config, data)
    return data_nan, data, lam


def exp_distributed(r_seed,m, n, mv_config, beta=None):
    
    np.random.seed(r_seed)
    
    print('='*50)
    print('|' + 'exp_ind_distributed'.center(48) + '|')
    print('|' + ('r_seed:'+ str(r_seed) + ', m:' + str(m) + ', n:' + str(n)).center(48) + '|')
    print('|' + str(mv_config).center(48) + '|')
    print('='*50)
    
    if beta is None:
        beta = np.random.random(n)
    
    data = np.random.exponential(beta, [m, n])

    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    
    plt.scatter(data[:, 0], data[:, 1])
    plt.show()
    data_nan, data = fillin_nan_MCAR(r_seed, mv_config, data)
    return data_nan, data, beta

def cat_distributed(r_seed, m, n, mv_config, rand_p_list=None):
    
    np.random.seed(r_seed)
    
    print('='*50)
    print('|' + 'cat_ind_distributed'.center(48) + '|')
    print('|' + ('r_seed:'+ str(r_seed) + ', m:' + str(m) + ', n:' + str(n)).center(48) + '|')
    print('|' + str(mv_config).center(48) + '|')
    print('='*50)
    
    cat_list = [2, 2, 5, 5, 10]
    data = np.zeros([m, n])
    
    if rand_p_list is None:
        rand_p_list = []
        for i in range(n):
            rand_p = np.random.rand(cat_list[i])
            rand_p = rand_p/rand_p.sum()
            rand_p_list.append(rand_p)
            
    for i in range(n):
        
        data[:, i] = np.random.choice(cat_list[i],  m, p=rand_p_list[i])

    data_nan, data = fillin_nan_MCAR(r_seed, mv_config, data)
    return data_nan, data, rand_p_list

def fillin_nan_MCAR(r_seed, mv_config, X):
    
    np.random.seed(r_seed)
    
    print('*'*5 + 'creating missing values'.center(40) + '*'*5)
    
    X_with_nan = np.array(X)
    m = X_with_nan.shape[0]
    for key in mv_config:
        mr = mv_config[key] # missing ratio
        num_mv = int(mr * m) # number of missing value
        mv_idx = np.random.permutation(m)[:num_mv] # random missing value index
        X_with_nan[mv_idx, key] = np.nan
    
    return X_with_nan, X

def fillin_nan_MNAR(r_seed, mv_config, X, mr_config):
    
    np.random.seed(r_seed)
    
    print('*'*5 + 'creating missing values'.center(40) + '*'*5)
    
    X_with_nan = np.array(X)
    m = X_with_nan.shape[0]
    for key in mv_config:
        mr = mv_config[key] # missing ratio
        num_mv = int(mr * m) # number of missing value
        mv_idx = np.random.permutation(m)[:num_mv] # random missing value index
        X_with_nan[mv_idx, key] = np.nan
    
    return X_with_nan, X


def load_realdata(r_seed, mv_config, num_file, main_folder):
    
    #main_folder = 'Data/Exp4_ReaDriftDetection/1_Higgs/Data/back_higg/'
    train = np.loadtxt(main_folder + 'data_train.csv', delimiter=',')
    train_with_nan, train= fillin_nan_MCAR(r_seed, mv_config, train)
    
    w0_with_nan_list = []
    w0_list = []
    w1_with_nan_list = []
    w1_list = []
    
    for i in range(num_file):
        
        w0_i = np.loadtxt(main_folder + 'data_w0_' + str(i) + '.csv', delimiter=',')
        w0_i_with_nan = fillin_nan_MCAR(r_seed, mv_config, w0_i)
        w1_i = np.loadtxt(main_folder + 'data_w1_' + str(i) + '.csv', delimiter=',')
        w1_i_with_nan = fillin_nan_MCAR(r_seed, mv_config, w1_i)
        
        w0_list.append(w0_i)
        w1_list.append(w1_i)
        w0_with_nan_list.append(w0_i_with_nan)
        w1_with_nan_list.append(w1_i_with_nan)
        
    return train_with_nan, train, w0_with_nan_list, w0_list, w1_with_nan_list, w1_list

