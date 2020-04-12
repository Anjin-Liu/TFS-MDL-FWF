# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 11:43:35 2019

@author: Anjin Liu
@email: anjin.liu@uts.edu.au
@affiliation: The Drift, DeSI, CAI, UTS   
"""

from mf_distance import data_handler as dh
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
#from sklearn.preprocessing.imputation import impute
from sklearn.impute import SimpleImputer
#from MFD_kMeans_lib import random_missing_train_set, mask_distance_vector, full_join_data
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
    train_Y = euclidean_distances(X_hasNan, X_hasNan)
    train_Y = train_Y.flatten()
    
    X_masked_hasNan = X_maskedNan[X_maskedNan_hasNan_idx, :]
    train_X = extract_feats_transform(X_masked_hasNan)
    
    return train_X, train_Y
    
    
def extract_feats_transform(X):
    
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
    return train_X
    
# by out join data
def distance_estimator_full_join(X_nan, X):
    
    X_noNan_idx = np.where(~np.isnan(X_nan).any(axis=1))[0]
    X_noNan = X_nan[X_noNan_idx]
        
    nan_ratio = (np.isnan(X_nan)*1).sum(axis=0)
    nan_ratio = nan_ratio/X_nan.shape[0]
    mv_config = {}
    for i in range(nan_ratio.shape[0]):
        mv_config[i] = nan_ratio[i]
        
    train_X, train_Y = full_join_data(X_noNan, mv_config)
    
    est = GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=4, random_state=0)
    train_X = np.nan_to_num(train_X)
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(train_X)
    train_X = imp.transform(train_X)
    est.fit(train_X, train_Y)
    
    
    pd_X_Nan = pd.DataFrame(X_nan)
    pd_X_Nan['key'] = 0
    all_merge = pd.merge(pd_X_Nan, pd_X_Nan, on='key', how='outer')
    all_merge = all_merge.drop(columns=['key'])
    all_merge = all_merge.values
    
    m = X.shape[0]
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(all_merge)
    all_merge = imp.transform(all_merge)
    est_dist_flatten = est.predict(all_merge)
    est_dist_mat = est_dist_flatten.reshape([m, m])
    est_dist_mat = non_missing_dist_filter(X_noNan, X_noNan_idx, est_dist_mat)
    
    act_dist_mat = euclidean_distances(X, X)
    
    
    dist_residual = est_dist_mat - act_dist_mat
    total_num = m*(m-1) # m-1 is for removing all 0 diagonal values (i-i distance) of the matrix
    
    dist_mae = np.abs(dist_residual).sum()/total_num
    dist_rmse = np.sqrt((dist_residual*dist_residual).sum()/total_num)
    
    return dist_mae, dist_rmse

#backup
def distance_estimator_uncertainty_mask(X_nan, X):
    
    X_noNan_idx = np.where(~np.isnan(X_nan).any(axis=1))[0]
    X_noNan = X_nan[X_noNan_idx]
        
    nan_ratio = (np.isnan(X_nan)*1).sum(axis=0)
    nan_ratio = nan_ratio/X_nan.shape[0]
    mv_config = {}
    for i in range(nan_ratio.shape[0]):
        mv_config[i] = nan_ratio[i]
        
    md = random_missing_train_set(X_noNan, mv_config)
    
    uncertain_degree = md[:, :-2]
    sim_dist = md[:, -2].reshape(-1,1)
    act_dist = md[:, -1].reshape(-1,1) 
    #dif_dist = sim_dist - act_dist
    
    masked_X = np.hstack([uncertain_degree, sim_dist])
    masked_Y = act_dist
    
    est = GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=4, random_state=0)
    est.fit(masked_X, masked_Y.ravel())
    
    m = X_nan.shape[0]
    est_dist_mat = np.zeros([m, m])
    for i in range(X_nan.shape[0] - 1):
        #print('predicting @ ' + str(i))
        for j in range(i + 1, X_nan.shape[0]):
            X_test = mask_distance_vector(X_nan[i,:], X_nan[j,:])
            Y_hat = est.predict(X_test.reshape(1, -1))
            est_dist_mat[i, j] = Y_hat
            est_dist_mat[j, i] = Y_hat
    
    est_dist_mat = non_missing_dist_filter(X_noNan, X_noNan_idx, est_dist_mat)
    act_dist_mat = euclidean_distances(X, X)
    
    dist_residual = est_dist_mat - act_dist_mat
    total_num = m*(m-1) # m-1 is for removing all 0 diagonal values (i-i distance) of the matrix
    
    dist_mae = np.abs(dist_residual).sum()/total_num
    dist_rmse = np.sqrt((dist_residual*dist_residual).sum()/total_num)
    
    return dist_mae, dist_rmse
#'''  
    
def non_missing_dist_filter(X_noNan, X_noNan_idx, est_dist_mat):
      
    X_noNan_dist_mat = euclidean_distances(X_noNan, X_noNan)
    replace_to_idx = np.array(np.meshgrid(X_noNan_idx, X_noNan_idx)).T.reshape(-1,2)
    temp_idx = np.arange(X_noNan.shape[0])
    replace_fr_idx = np.array(np.meshgrid(temp_idx, temp_idx)).T.reshape(-1,2)
    est_dist_mat[replace_to_idx[:,0], replace_to_idx[:,1]] = X_noNan_dist_mat[replace_fr_idx[:,0], replace_fr_idx[:,1]]
    np.fill_diagonal(est_dist_mat, 0)
    
    return est_dist_mat

def exp_distance_estimation(data_nan, data):
    
    rae_zero_imput, rmse_zero_imput = imputation_simp(data_nan, data, 'constant', 0)
    rae_mean_imput, rmse_mean_imput = imputation_simp(data_nan, data, 'mean')
    rae_medi_imput, rmse_medi_imput = imputation_simp(data_nan, data, 'median')
    rae_mfre_imput, rmse_mfre_imput = imputation_simp(data_nan, data, 'most_frequent')
    rae_iter_imput, rmse_iter_imput = imputation_iter(data_nan, data)
    
    rae_destEF_imput, rmse_destFE_imput, feats_imp = distance_estimator_extracted_feats(data_nan, data)
    
    print('rae_zero_imput, %.5f, rmse_zero_imput, %.5f' % (rae_zero_imput, rmse_zero_imput))
    print('rae_mean_imput, %.5f, rmse_mean_imput, %.5f' % (rae_mean_imput, rmse_mean_imput))
    print('rae_medi_imput, %.5f, rmse_medi_imput, %.5f' % (rae_medi_imput, rmse_medi_imput))
    print('rae_mfre_imput, %.5f, rmse_mfre_imput, %.5f' % (rae_mfre_imput, rmse_mfre_imput))
    print('rae_iter_imput, %.5f, rmse_iter_imput, %.5f' % (rae_iter_imput, rmse_iter_imput))
    
    print('rae_destEF_imput, %.5f, rmse_destFE_imput, %.5f' % (rae_destEF_imput, rmse_destFE_imput))
    
    result = [rae_zero_imput, rmse_zero_imput,
        rae_mean_imput, rmse_mean_imput,
        rae_medi_imput, rmse_medi_imput,
        rae_mfre_imput, rmse_mfre_imput,
        rae_iter_imput, rmse_iter_imput,
        rae_destEF_imput, rmse_destFE_imput]
    
    return result, feats_imp
  
def print_results(result_array, title):
    
    result_mean = np.mean(result_array, axis=0)
    result_std = np.std(result_array, axis=0)
    
    print('')
    print(' Results ' + title)
    print(', MAE_mean, MAE_std, RMSE_mean, RMSE_std')
    print('zero , %.5f, %.5f, %.5f, %.5f' %(result_mean[0], result_std[0], result_mean[1], result_std[1]))
    print('mean , %.5f, %.5f, %.5f, %.5f' %(result_mean[2], result_std[2], result_mean[3], result_std[3]))
    print('medi , %.5f, %.5f, %.5f, %.5f' %(result_mean[4], result_std[4], result_mean[5], result_std[5]))
    print('mfre , %.5f, %.5f, %.5f, %.5f' %(result_mean[6], result_std[6], result_mean[7], result_std[7]))
    print('iter , %.5f, %.5f, %.5f, %.5f' %(result_mean[8], result_std[8], result_mean[9], result_std[9]))
    print('mask , %.5f, %.5f, %.5f, %.5f' %(result_mean[10], result_std[10], result_mean[11], result_std[11]))
    
def print_feat_imp(feat_imp_array, title):
    
    feat_imp_mean = np.mean(feat_imp_array, axis=0)
    feat_imp_std = np.std(feat_imp_array, axis=0)
    
    print()
    print(' Feature Importance ' + title)
    print('Feat_id, imp_mean, imp_std')
    for i in range(feat_imp_mean.shape[0]):
        print('Feat-%d, %.5f, %.5f' % (i, feat_imp_mean[i], feat_imp_std[i]))
        
def plot_feat_imp(all_feat_imp):
    
    fig = plt.figure(1, figsize = (8,10))
    
    label_list = []
    for i in range(1, 21):
        label_list.append('f'+str(i))
    label_list = label_list + ['zero', 'mean', 'median', 'mfre', 'iter']
    label_list = np.array(label_list)
    
    ax1 = plt.subplot2grid((4, 2), (0, 0), colspan=2)
    plt.title('Uniform')
    feat_imp_mean = all_feat_imp[0].mean(axis=0)
    feat_imp_std = all_feat_imp[0].std(axis=0)
    indices = np.argsort(-feat_imp_mean)
    plt.bar(range(25), feat_imp_mean[indices], color="r", yerr=feat_imp_std[indices], align="center")
    plt.xticks(range(25), label_list[indices])
    plt.setp(ax1.get_xticklabels(), rotation=45, horizontalalignment='right')
    plt.xlim([-1, 25])
    
    ax2 = plt.subplot2grid((4, 2), (1, 0), colspan=2)
    plt.title('Gaussian')
    feat_imp_mean = all_feat_imp[1].mean(axis=0)
    feat_imp_std = all_feat_imp[1].std(axis=0)
    indices = np.argsort(-feat_imp_mean)
    plt.bar(range(25), feat_imp_mean[indices], color="r", yerr=feat_imp_std[indices], align="center")
    plt.xticks(range(25), label_list[indices])
    plt.setp(ax2.get_xticklabels(), rotation=45, horizontalalignment='right')
    plt.xlim([-1, 25])
    
    ax3 = plt.subplot2grid((4, 2), (2, 0), colspan=2)
    plt.title('Exponential')
    feat_imp_mean = all_feat_imp[2].mean(axis=0)
    feat_imp_std = all_feat_imp[2].std(axis=0)
    indices = np.argsort(-feat_imp_mean)
    plt.bar(range(25), feat_imp_mean[indices], color="r", yerr=feat_imp_std[indices], align="center")
    plt.xticks(range(25), label_list[indices])
    plt.setp(ax3.get_xticklabels(), rotation=45, horizontalalignment='right')
    plt.xlim([-1, 25])
    
    label_list = []
    for i in range(1, 11):
        label_list.append('f'+str(i))
    label_list = label_list + ['zero', 'mean', 'median', 'mfre', 'iter'] 
    label_list = np.array(label_list)
    
    ax4 = plt.subplot2grid((4, 2), (3, 0))
    plt.title('Poission')
    feat_imp_mean = all_feat_imp[3].mean(axis=0)
    feat_imp_std = all_feat_imp[3].std(axis=0)
    indices = np.argsort(-feat_imp_mean)
    plt.bar(range(15), feat_imp_mean[indices], color="r", yerr=feat_imp_std[indices], align="center")
    plt.xticks(range(15), label_list[indices])
    plt.setp(ax4.get_xticklabels(), rotation=45, horizontalalignment='right')
    plt.xlim([-1, 15])
    
    ax5 = plt.subplot2grid((4, 2), (3, 1))
    plt.title('Categorical')
    feat_imp_mean = all_feat_imp[4].mean(axis=0)
    feat_imp_std = all_feat_imp[4].std(axis=0)
    indices = np.argsort(-feat_imp_mean)
    plt.bar(range(15), feat_imp_mean[indices], color="r", yerr=feat_imp_std[indices], align="center")
    plt.xticks(range(15), label_list[indices])
    plt.setp(ax5.get_xticklabels(), rotation=45, horizontalalignment='right')
    plt.xlim([-1, 15])
    
    plt.tight_layout()
    plt.show()
    
    fig.savefig("Results/Exp_1_feat_imp.pdf", bbox_inches='tight')
    
if __name__ == "__main__":
    
    r_seed = 1

    mv_config = {}
    for i in range(5):
        mv_config[i] = 0.2
        
    num_run = 2

    result_uni = np.zeros([num_run, 12])
    feat_imp_uni = np.zeros([num_run, 10*2+5])
    for i in range(num_run):
        r_seed = i
        data_nan, data = dh.uni_distributed(r_seed, 500, 10, mv_config)
        result_uni[i, :], feat_imp_uni[i, :] = exp_distance_estimation(data_nan, data)
       
    result_gau = np.zeros([num_run, 12])
    feat_imp_gau = np.zeros([num_run, 10*2+5])
    mu_sigma = None
    for i in range(num_run):
        r_seed = i
        data_nan, data, mu_sigma = dh.gau_distributed(r_seed, 500, 10, mv_config, mu_sigma)
        result_gau[i, :], feat_imp_gau[i, :] = exp_distance_estimation(data_nan, data)
    
    result_exp = np.zeros([num_run, 12])
    feat_imp_exp = np.zeros([num_run, 10*2+5])
    beta = None
    for i in range(num_run):
        r_seed = i
        data_nan, data, beta = dh.exp_distributed(r_seed, 500, 10, mv_config, beta)
        result_exp[i, :], feat_imp_exp[i, :] = exp_distance_estimation(data_nan, data)
        
    result_poi = np.zeros([num_run, 12])
    feat_imp_poi = np.zeros([num_run, 5*2+5])
    lam = None
    for i in range(num_run):
        r_seed = i
        data_nan, data, lam = dh.poi_distributed(r_seed, 500, 5, mv_config, lam)
        result_poi[i, :], feat_imp_poi[i, :] = exp_distance_estimation(data_nan, data)
      
    result_cat = np.zeros([num_run, 12])
    feat_imp_cat = np.zeros([num_run, 5*2+5])
    rand_p_list = None
    for i in range(num_run):
        r_seed = i
        data_nan, data, rand_p_list = dh.cat_distributed(r_seed, 500, 5, mv_config, rand_p_list)
        result_cat[i, :], feat_imp_cat[i, :] = exp_distance_estimation(data_nan, data)
    
    print_results(result_uni, 'uni_ind_distributed')
    print_results(result_gau, 'gau_ind_distributed')
    print_results(result_exp, 'exp_ind_distributed')
    print_results(result_poi, 'poi_ind_distributed')
    print_results(result_cat, 'cat_ind_distributed')
    
    all_results = [result_uni, result_gau, result_exp, result_poi, result_cat]
    all_feat_imp = [feat_imp_uni, feat_imp_gau, feat_imp_exp, feat_imp_poi, feat_imp_cat]
    plot_feat_imp(all_feat_imp)
    
    
    
    
    
    