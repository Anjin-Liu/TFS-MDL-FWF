# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 11:43:35 2019

@author: 119275
"""

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer

def imputation_iter(X_nan, X):
    
    imp = IterativeImputer(max_iter=10, random_state=0)
    imp.fit(X_nan)
    iter_imput_X = imp.transform(X_nan)
    
    act_dist_mat = euclidean_distances(X, X)
    est_dist_mat = euclidean_distances(iter_imput_X, iter_imput_X)
    
    dist_residual = est_dist_mat - act_dist_mat
    m = X_nan.shape[0]
    total_num = m*(m-1) # m-1 is for removing all 0 diagonal values (i-i distance) of the matrix
    
    dist_mae = np.abs(dist_residual).sum()/total_num
    dist_rmse = np.sqrt((dist_residual*dist_residual).sum()/total_num)
    
    return dist_mae, dist_rmse
#'''
  
def imputation_simp(X_nan, X, strategy='mean', fill_value=None):
    
    imp = SimpleImputer(missing_values=np.nan, strategy=strategy, fill_value=fill_value)
    imp.fit(X_nan)
    iter_imput_X = imp.transform(X_nan)
    
    act_dist_mat = euclidean_distances(X, X)
    est_dist_mat = euclidean_distances(iter_imput_X, iter_imput_X)
    
    dist_residual = est_dist_mat - act_dist_mat
    m = X_nan.shape[0]
    total_num = m*(m-1) # m-1 is for removing all 0 diagonal values (i-i distance) of the matrix
    
    dist_mae = np.abs(dist_residual).sum()/total_num
    dist_rmse = np.sqrt((dist_residual*dist_residual).sum()/total_num)
    
    return dist_mae, dist_rmse


#'''
# by out join data
def distance_estimator_extracted_feats(X_nan, X):
    
    X_noNan_idx = np.where(~np.isnan(X_nan).any(axis=1))[0]
    X_noNan = X_nan[X_noNan_idx]

    #========================#
    # estiamte missing ratio #
    #========================#
    nan_ratio = (np.isnan(X_nan)*1).sum(axis=0)
    nan_ratio = nan_ratio/X_nan.shape[0]
    mv_config = {}
    for i in range(nan_ratio.shape[0]):
        mv_config[i] = nan_ratio[i]
    
    train_X, train_Y = extract_feats_masking(X_noNan, mv_config)
    
    est = GradientBoostingRegressor(loss='huber', n_estimators=500, learning_rate=0.05, max_depth=4, random_state=0)
    est.fit(train_X, train_Y)
    
    #print(est.feature_importances_)
    
    test_X = extract_feats_transform(X_nan)
    est_dist_flatten = est.predict(test_X)
    m = X.shape[0]
    est_dist_mat = est_dist_flatten.reshape([m, m])
    est_dist_mat = non_missing_dist_filter(X_noNan, X_noNan_idx, est_dist_mat)
    act_dist_mat = euclidean_distances(X, X)
    dist_residual = est_dist_mat - act_dist_mat
    total_num = m*(m-1) # m-1 is for removing all 0 diagonal values (i-i distance) of the matrix
    
    dist_mae = np.abs(dist_residual).sum()/total_num
    dist_rmse = np.sqrt((dist_residual*dist_residual).sum()/total_num)
    
    return dist_mae, dist_rmse, est.feature_importances_
    
def extract_feats_masking(X_noNan, mv_config):
    
    m = X_noNan.shape[0]
    X_maskedNan = np.array(X_noNan)
    for key in mv_config:
        mr = mv_config[key] # missing ratio
        num_mv = int(mr * m)
        mv_idx = np.random.permutation(m)[:num_mv] # w0 random missing value index
        X_maskedNan[mv_idx, key] = np.nan
        
    #========================================#
    # generatre mask-distance matrix         #
    # 0 no missing value                     #
    # 1 one x_ik or x_jk has a missing value #
    # both x_ik and x_jk are missing         #
    # at the end is the distance without     #
    # considering missing values             #
    #========================================#
    X_maskedNan_hasNan_idx = np.isnan(X_maskedNan).any(axis=1)
    
    X_hasNan = X_noNan[X_maskedNan_hasNan_idx, :]
    #if X_hasNan.shape[0] < 50:
    #    return None, None
    train_Y = euclidean_distances(X_hasNan, X_hasNan)
    train_Y = train_Y.flatten()
    
    X_masked_hasNan = X_maskedNan[X_maskedNan_hasNan_idx, :]
    train_X = extract_feats_transform(X_masked_hasNan)
    
    return train_X, train_Y
    
    
def extract_feats_transform(X, Y=None):
    
    if Y is None:
        
        #==================================#
        # use impute distances as features #
        #==================================#
        imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
        zero_imput = imp.fit_transform(X)
        zero_imput = euclidean_distances(zero_imput, zero_imput)
        zero_imput = zero_imput.flatten().reshape(-1, 1)
        
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        mean_imput = imp.fit_transform(X)
        mean_imput = euclidean_distances(mean_imput, mean_imput)
        mean_imput = mean_imput.flatten().reshape(-1, 1)
        
        imp = SimpleImputer(missing_values=np.nan, strategy='median')
        medi_imput = imp.fit_transform(X)
        medi_imput = euclidean_distances(medi_imput, medi_imput)
        medi_imput = medi_imput.flatten().reshape(-1, 1)
        
        imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        mfre_imput = imp.fit_transform(X)
        mfre_imput = euclidean_distances(mfre_imput, mfre_imput)
        mfre_imput = mfre_imput.flatten().reshape(-1, 1)
        
        imp = IterativeImputer(max_iter=10, random_state=0)
        iter_imput = imp.fit_transform(X)
        iter_imput = euclidean_distances(iter_imput, iter_imput)
        iter_imput = iter_imput.flatten().reshape(-1, 1)
        
        #=============================#
        # missing value masked vector #
        #=============================#
        X_masked_hasNan_masked_vector = np.isnan(X)*1
        pd_X_Nan = pd.DataFrame(X_masked_hasNan_masked_vector)
        pd_X_Nan['key'] = 0
        all_merge = pd.merge(pd_X_Nan, pd_X_Nan, on='key', how='outer')
        all_merge = all_merge.drop(columns=['key'])
        all_merge = all_merge.values
        train_X = np.hstack([all_merge, zero_imput, mean_imput, medi_imput, mfre_imput, iter_imput])
        
    else:
        
        #==================================#
        # use impute distances as features #
        #==================================#
        imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
        zero_imput_X = imp.fit_transform(X)
        zero_imput_Y = imp.fit_transform(Y)
        zero_imput = euclidean_distances(zero_imput_X, zero_imput_Y)
        zero_imput = zero_imput.flatten().reshape(-1, 1)
        
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        mean_imput_X = imp.fit_transform(X)
        mean_imput_Y = imp.fit_transform(Y)
        mean_imput = euclidean_distances(mean_imput_X, mean_imput_Y)
        mean_imput = mean_imput.flatten().reshape(-1, 1)
        
        imp = SimpleImputer(missing_values=np.nan, strategy='median')
        medi_imput_X = imp.fit_transform(X)
        medi_imput_Y = imp.fit_transform(Y)
        medi_imput = euclidean_distances(medi_imput_X, medi_imput_Y)
        medi_imput = medi_imput.flatten().reshape(-1, 1)
        
        imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        mfre_imput_X = imp.fit_transform(X)
        mfre_imput_Y = imp.fit_transform(Y)
        mfre_imput = euclidean_distances(mfre_imput_X, mfre_imput_Y)
        mfre_imput = mfre_imput.flatten().reshape(-1, 1)
        
        imp = IterativeImputer(max_iter=10, random_state=0)
        iter_imput_X = imp.fit_transform(X)
        iter_imput_Y = imp.fit_transform(Y)
        iter_imput = euclidean_distances(iter_imput_X, iter_imput_Y)
        iter_imput = iter_imput.flatten().reshape(-1, 1)
        
        #=============================#
        # missing value masked vector #
        #=============================#
        X_masked_hasNan_masked_vector = np.isnan(X)*1
        Y_masked_hasNan_masked_vector = np.isnan(Y)*1
        pd_X_Nan = pd.DataFrame(X_masked_hasNan_masked_vector)
        pd_Y_Nan = pd.DataFrame(Y_masked_hasNan_masked_vector)
        pd_X_Nan['key'] = 0
        pd_Y_Nan['key'] = 0
        all_merge = pd.merge(pd_X_Nan, pd_Y_Nan, on='key', how='outer')
        all_merge = all_merge.drop(columns=['key'])
        all_merge = all_merge.values
        train_X = np.hstack([all_merge, zero_imput, mean_imput, medi_imput, mfre_imput, iter_imput])
        
    return train_X 
    
def non_missing_dist_filter(X_noNan, X_noNan_idx, est_dist_mat):
      
    X_noNan_dist_mat = euclidean_distances(X_noNan, X_noNan)
    replace_to_idx = np.array(np.meshgrid(X_noNan_idx, X_noNan_idx)).T.reshape(-1,2)
    temp_idx = np.arange(X_noNan.shape[0])
    replace_fr_idx = np.array(np.meshgrid(temp_idx, temp_idx)).T.reshape(-1,2)
    est_dist_mat[replace_to_idx[:,0], replace_to_idx[:,1]] = X_noNan_dist_mat[replace_fr_idx[:,0], replace_fr_idx[:,1]]
    np.fill_diagonal(est_dist_mat, 0)
    
    return est_dist_mat
    
    
    


