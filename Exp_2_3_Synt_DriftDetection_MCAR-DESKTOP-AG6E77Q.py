# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 11:43:35 2019

@author: Anjin Liu
@email: anjin.liu@uts.edu.au
@affiliation: The Drift, DeSI, CAI, UTS   
 
"""

from mf_distance import data_handler as dh
import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer

from detection_methods import mul_wald_test
import scipy.stats as stats

from detection_methods import kmeans_chi2_test
from mf_distance import mf_distance_kmeans_chi2_test

import freqopttest.tst as tst
from freqopttest.data import TSTData
from freqopttest import kernel
from detection_methods import libquanttree as qt
from sklearn import metrics
def perform_mmd_test(train_miss_impute, test_miss_impute, train_full, test_full, alpha, 
                     mmd_miss_impute=None, mmd_full=None):
    
    mmd_result = np.zeros(2)
    
    sb_data_miss_impute = TSTData(train_miss_impute, test_miss_impute)
    if mmd_miss_impute is None:
        print('ini')
        x,y = sb_data_miss_impute.xy()
        dist_mat_miss_impute = metrics.pairwise_distances(x, y)
        the_kernel = kernel.KGauss(dist_mat_miss_impute.std())
        mmd_miss_impute = tst.QuadMMDTest(the_kernel, alpha=alpha)
    test_result = mmd_miss_impute.perform_test(sb_data_miss_impute)
    if test_result['h0_rejected']:
        mmd_result[0] = 1
    
    sb_data_full = TSTData(train_full, test_full)
    if mmd_full is None:
        x,y = sb_data_full.xy()
        dist_mat_full = metrics.pairwise_distances(x, y)
        the_kernel = kernel.KGauss(dist_mat_full.std())
        mmd_full = tst.QuadMMDTest(the_kernel, alpha=alpha)
    test_result = mmd_full.perform_test(sb_data_full)
    if test_result['h0_rejected']:
        mmd_result[1] = 1
        
    return mmd_result, mmd_miss_impute, mmd_full

def perform_me_test(train_miss_impute, test_miss_impute, train_full, test_full, alpha, 
                     test_locs_miss=None, gwidth_miss=None, test_locs_full=None, gwidth_full=None):
    me_result = np.zeros(2)
    
    op = {
    'n_test_locs': 10, # number of test locations to optimize
    'max_iter': 200, # maximum number of gradient ascent iterations
    'locs_step_size': 1.0, # step size for the test locations (features)
    'gwidth_step_size': 0.1, # step size for the Gaussian width
    'tol_fun': 1e-4, # stop if the objective does not increase more than this.
    'seed': 0  # random seed
    }
    
    sb_data_miss_impute = TSTData(train_miss_impute, test_miss_impute)
    train_miss_impute_sb, dumy = sb_data_miss_impute.split_tr_te(tr_proportion=1, seed=1)
    dumy, test_miss_impute_sb = sb_data_miss_impute.split_tr_te(tr_proportion=0, seed=1)
    #half_size = int(train_miss_impute.shape[0]/2)
    #train_miss_impute_sb = TSTData(train_miss_impute[:half_size], train_miss_impute[half_size:half_size*2])
    #test_miss_impute_sb = TSTData(train_miss_impute, test_miss_impute)
    
    if test_locs_miss is None:
        test_locs_miss, gwidth_miss, info = tst.MeanEmbeddingTest.optimize_locs_width(train_miss_impute_sb, alpha, **op)
    met_opt = tst.MeanEmbeddingTest(test_locs_miss, gwidth_miss, alpha)
    test_result = met_opt.perform_test(test_miss_impute_sb)
    if test_result['h0_rejected']:
        me_result[0] = 1
    
    sb_data_full = TSTData(train_full, test_full)
    train_full_sb, dumy = sb_data_full.split_tr_te(tr_proportion=1, seed=1)
    dumy, test_full_sb = sb_data_full.split_tr_te(tr_proportion=0, seed=1)
    
    if test_locs_full is None:
        test_locs_full, gwidth_full, info = tst.MeanEmbeddingTest.optimize_locs_width(train_full_sb, alpha, **op)
    met_opt = tst.MeanEmbeddingTest(test_locs_full, gwidth_full, alpha)
    test_result = met_opt.perform_test(test_full_sb)
    if test_result['h0_rejected']:
        me_result[1] =  1
        
    return me_result, test_locs_miss, gwidth_miss, test_locs_full, gwidth_full

def perform_QTree_test(train_miss_impute, test_miss_impute, train_full, test_full, alpha, Qtree_Htest_miss_impute=None, Qtree_Htest_full=None):
    
    QTree_result = np.zeros(2)
    
    m1 = train_miss_impute.shape[0]
    m2 = test_miss_impute.shape[0]
    K = int(m1/50)
    B = 5000
    
    if Qtree_Htest_miss_impute is None:
        qtree_miss_impute = qt.QuantTree(K)
        qtree_miss_impute.build_histogram(train_miss_impute, True)
        Qtree_Htest_miss_impute = qt.ChangeDetectionTest(qtree_miss_impute, m2, qt.pearson_statistic)
        threshold = qt.ChangeDetectionTest.get_precomputed_quanttree_threshold('pearson', alpha, K, m1, m2)
        if threshold is None:
            threshold = Qtree_Htest_miss_impute.estimate_quanttree_threshold(alpha, B)
            Qtree_Htest_miss_impute.set_threshold(alpha, threshold)
            print('pearson_dist_free', m1, m2, threshold)
    
    hp, _ = Qtree_Htest_miss_impute.reject_null_hypothesis(test_miss_impute, alpha)
    if hp:
        QTree_result[0] = 1
        
    if Qtree_Htest_full is None:
        qtree_full = qt.QuantTree(K)
        qtree_full.build_histogram(train_full, True)
        Qtree_Htest_full = qt.ChangeDetectionTest(qtree_full, m2, qt.pearson_statistic)
        threshold = qt.ChangeDetectionTest.get_precomputed_quanttree_threshold('pearson', alpha, K, m1, m2)
        if threshold is None:
            threshold = Qtree_Htest_full.estimate_quanttree_threshold(alpha, B)
            Qtree_Htest_full.set_threshold(alpha, threshold)
            print('pearson_dist_free', m1, m2, threshold)
    
    hp, _ = Qtree_Htest_full.reject_null_hypothesis(test_full, alpha)
    if hp:
        QTree_result[1] = 1
    
    return QTree_result, Qtree_Htest_miss_impute, Qtree_Htest_full
    
def perform_mww_test(train_miss_impute, test_miss_impute, train_full, test_full, alpha):
    
    mww_result = np.zeros(2)
    
    W_miss, R_miss = mul_wald_test.ww_test(train_miss_impute, test_miss_impute)
    pvalue_miss = stats.norm.cdf(W_miss)  # one sided test
    reject_miss = pvalue_miss <= alpha
    if reject_miss:
        mww_result[0] = 1
    
    W_full, R_full = mul_wald_test.ww_test(train_full, test_full)
    pvalue_full = stats.norm.cdf(W_full)  # one sided test
    reject_full = pvalue_full <= alpha
    if reject_full:
        mww_result[1] =  1
        
    return mww_result

def perform_kmean_chi2_test(train_miss_impute, test_miss_impute, train_full, test_full, alpha, kchi2_miss_impute=None, kchi2_full=None):
    
    m = train_miss_impute.shape[0]
    kmean_chi2_result = np.zeros(2)
    
    if kchi2_miss_impute is None:
        kchi2_miss_impute = kmeans_chi2_test.kMeansChi2(int(m/50))
        kchi2_miss_impute.buildkMeans(train_miss_impute)
        
    if kchi2_full is None:
        kchi2_full = kmeans_chi2_test.kMeansChi2(int(m/50))
        kchi2_full.buildkMeans(train_full)
    
    kmean_chi2_result[0] = kchi2_miss_impute.drift_detection(test_miss_impute, alpha)
    kmean_chi2_result[1] = kchi2_full.drift_detection(test_full, alpha)
    
    return kmean_chi2_result, kchi2_miss_impute, kchi2_full

def perform_mfkmean_chi2_test(train_miss, test_miss, train_full, test_full, alpha, mfkchi2_miss=None, mfkchi2_full=None, apply_fuzzy='Rectangle'):
    
    m = train_miss.shape[0]
    mfkmean_chi2_result = np.zeros(2)
    
    if mfkchi2_miss is None:
        mfkchi2_miss = mf_distance_kmeans_chi2_test.MFkMeansChi2(int(m/50), apply_fuzzy)
        mfkchi2_miss.buildMFkMeans(train_miss)
        
    if mfkchi2_full is None:
        mfkchi2_full = mf_distance_kmeans_chi2_test.MFkMeansChi2(int(m/50), apply_fuzzy)
        mfkchi2_full.buildMFkMeans(train_full)
    
    mfkmean_chi2_result[0] = mfkchi2_miss.drift_detection(test_miss, alpha)
    mfkmean_chi2_result[1] = mfkchi2_full.drift_detection(test_full, alpha)
    
    return mfkmean_chi2_result, mfkchi2_miss, mfkchi2_full

def mcar_drift_detection(run_method_list, drift_type='gau_mean', alpha=0.05, num_test_PerDriftDelta=150):
    
    m = 500
    n = 10
    mv_config = {}
    for i in range(5):
        mv_config[i] = 0.2
    
    if drift_type=='uni_mean':
        train_drift_config = 0
        train_miss, train_full = dh.uni_distributed(0, m, n, mv_config, train_drift_config)
        imp = IterativeImputer(max_iter=10, random_state=0)
        train_miss_impute = imp.fit_transform(train_miss)
        drift_delta_max = 0.15
    elif drift_type=='gau_mean':
        train_miss, train_full, mu_sigma = dh.gau_distributed(0, m, n, mv_config)
        imp = IterativeImputer(max_iter=10, random_state=0)
        train_miss_impute = imp.fit_transform(train_miss)
        drift_delta_max = 0.35
    elif drift_type=='gau_cov':
        train_miss, train_full, mu_sigma = dh.gau_distributed(0, m, n, mv_config)
        imp = IterativeImputer(max_iter=10, random_state=0)
        train_miss_impute = imp.fit_transform(train_miss)
        drift_delta_max = 0.8
    elif drift_type=='poi_mean':
        train_miss, train_full, lam = dh.poi_distributed(0, m, n, mv_config)
        imp = IterativeImputer(max_iter=10, random_state=0)
        train_miss_impute = imp.fit_transform(train_miss)
        drift_delta_max = 2.5
    elif drift_type=='poi_rho':
        train_miss, train_full, lam = dh.poi_distributed(0, m, n, mv_config, rho=0)
        imp = IterativeImputer(max_iter=10, random_state=0)
        train_miss_impute = imp.fit_transform(train_miss)
        drift_delta_max = 0.8
        
    drift_delta_size = 10
    drift_delta = np.arange(0, drift_delta_max + drift_delta_max/drift_delta_size, drift_delta_max/drift_delta_size)
    detection_result = np.zeros([drift_delta_size+1, len(run_method_list)*2+1])
    detection_result[:, 0] = drift_delta
    
    delta_idx = 0
    kchi2_miss_impute = None
    kchi2_full = None
    
    mfkchi2_miss_crisp = None
    mfkchi2_full_crisp = None
    
    mfkchi2_miss_fuzzy_gau = None
    mfkchi2_full_fuzzy_gau = None
    
    mfkchi2_miss_fuzzy_tri = None
    mfkchi2_full_fuzzy_tri = None
    
    me_ml = None
    me_mg = None
    me_fl = None
    me_fg = None
        
    mmd_miss_impute = None
    mmd_full = None
    
    Qtree_Htest_miss_impute = None
    Qtree_Htest_full = None
    for delta in drift_delta:

        for i in range(num_test_PerDriftDelta):
            r_seed = delta_idx * num_test_PerDriftDelta + i
            
            if drift_type=='uni_mean':
                test_miss, test_full = dh.uni_distributed(r_seed, m, n, mv_config, drift_config=delta)
            elif drift_type=='gau_mean':
                test_miss, test_full, mu_sigma = dh.gau_distributed(r_seed, m, n, mv_config, mu_sigma, drift_config=('mean', delta))
            elif drift_type=='gau_cov':
                test_miss, test_full, mu_sigma = dh.gau_distributed(r_seed, m, n, mv_config, mu_sigma, drift_config=('cov', delta))
            elif drift_type=='poi_mean':
                test_miss, test_full, lam = dh.poi_distributed(r_seed, m, n, mv_config, lam, drift_config=delta)
            elif drift_type=='poi_rho':
                test_miss, test_full, lam = dh.poi_distributed(r_seed, m, n, mv_config, lam, rho=delta)
                
            test_miss_impute = imp.fit_transform(test_miss)
            
            run_method_counter = 0
            if 'mww' in run_method_list:
                mww_result = perform_mww_test(train_miss_impute, test_miss_impute, train_full, test_full, alpha)
                detection_result[delta_idx, 1+run_method_counter:3+run_method_counter] = detection_result[delta_idx, 1+run_method_counter:3+run_method_counter] + mww_result
                run_method_counter = run_method_counter + 2
                
            if 'kchi2' in run_method_list:
                kchi2_result, kchi2_miss_impute, kchi2_full = perform_kmean_chi2_test(
                        train_miss_impute, test_miss_impute, train_full, test_full, alpha, kchi2_miss_impute, kchi2_full)
                detection_result[delta_idx, 1+run_method_counter:3+run_method_counter] = detection_result[delta_idx, 1+run_method_counter:3+run_method_counter] + kchi2_result
                run_method_counter = run_method_counter + 2
                
            if 'mfkchi2_fuzzy_gau' in run_method_list:
                mfkchi2_fuzzy_result, mfkchi2_miss_fuzzy_gau, mfkchi2_full_fuzzy_gau = perform_mfkmean_chi2_test(
                        train_miss, test_miss, train_full, test_full, alpha, mfkchi2_miss_fuzzy_gau, mfkchi2_full_fuzzy_gau, apply_fuzzy='Gaussion')
                detection_result[delta_idx, 1+run_method_counter:3+run_method_counter] = detection_result[delta_idx, 1+run_method_counter:3+run_method_counter] + mfkchi2_fuzzy_result
                run_method_counter = run_method_counter + 2
                
            if 'mfkchi2_fuzzy_tri' in run_method_list:
                mfkchi2_fuzzy_result, mfkchi2_miss_fuzzy_tri, mfkchi2_full_fuzzy_tri = perform_mfkmean_chi2_test(
                        train_miss, test_miss, train_full, test_full, alpha, mfkchi2_miss_fuzzy_tri, mfkchi2_full_fuzzy_tri, apply_fuzzy='Triangle')
                detection_result[delta_idx, 1+run_method_counter:3+run_method_counter] = detection_result[delta_idx, 1+run_method_counter:3+run_method_counter] + mfkchi2_fuzzy_result
                run_method_counter = run_method_counter + 2
                
            if 'mfkchi2_crisp' in run_method_list:
                mfkchi2_crisp_result, mfkchi2_miss_crisp, mfkchi2_full_crisp = perform_mfkmean_chi2_test(
                        train_miss, test_miss, train_full, test_full, alpha, mfkchi2_miss_crisp, mfkchi2_full_crisp, apply_fuzzy='Crisp')
                detection_result[delta_idx, 1+run_method_counter:3+run_method_counter] = detection_result[delta_idx, 1+run_method_counter:3+run_method_counter] + mfkchi2_crisp_result
                run_method_counter = run_method_counter + 2
                
            if 'MMD' in run_method_list:
                mmd_result, mmd_miss_impute, mmd_full = perform_mmd_test(
                        train_miss_impute, test_miss_impute, train_full, test_full, alpha, mmd_miss_impute, mmd_full)
                print(mmd_result)
                detection_result[delta_idx, 1+run_method_counter:3+run_method_counter] = detection_result[delta_idx, 1+run_method_counter:3+run_method_counter] + mmd_result
                run_method_counter = run_method_counter + 2
            
            if 'ME' in run_method_list:
                me_result, me_ml, me_mg, me_fl, me_fg = perform_me_test(
                        train_miss_impute, test_miss_impute, train_full, test_full, alpha, me_ml, me_mg, me_fl, me_fg)
                detection_result[delta_idx, 1+run_method_counter:3+run_method_counter] = detection_result[delta_idx, 1+run_method_counter:3+run_method_counter] + me_result
                run_method_counter = run_method_counter + 2
            
            if 'QuantTree' in run_method_list:
                
                Qtree_result, Qtree_Htest_miss_impute, Qtree_Htest_full = perform_QTree_test(
                        train_miss_impute, test_miss_impute, train_full, test_full, alpha, Qtree_Htest_miss_impute, Qtree_Htest_full)
                detection_result[delta_idx, 1+run_method_counter:3+run_method_counter] = detection_result[delta_idx, 1+run_method_counter:3+run_method_counter] + Qtree_result
                run_method_counter = run_method_counter + 2
                
        delta_idx = delta_idx + 1
        
    return detection_result

if __name__ == "__main__":
    
    print('Exp3')
    num_test_PerDriftDelta = 20
    alpha = 0.05
    run_method_list = ['mww', 'kchi2', 'mfkchi2_fuzzy_gau', 'mfkchi2_fuzzy_tri', 'mfkchi2_crisp', 'ME', 'MMD', 'QuantTree']
    #run_method_list = ['ME']
    dataset_list = ['uni_mean', 'gau_mean', 'gau_cov', 'poi_mean', 'poi_rho']
    #dataset_list =  ['uni_mean', 'poi_mean', 'poi_rho']
    top_k=3
    
    uni_columns = ['delta']
    for i in range(len(run_method_list)):
        uni_columns.append(run_method_list[i]+'_miss')
        uni_columns.append(run_method_list[i]+'_full')
        
    for drift_type in dataset_list:
        detection_result = mcar_drift_detection(run_method_list, drift_type, alpha, num_test_PerDriftDelta)
        detection_result = pd.DataFrame(detection_result, columns=uni_columns)
        detection_result.to_csv('Results/'+drift_type+'_detection.csv', index=False)
    
    
    
    
    
    
















