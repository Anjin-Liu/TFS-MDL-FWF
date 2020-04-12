# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 11:43:35 2019

@author: 119275
"""

from mf_distance import data_handler as dh
import numpy as np
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
        x,y = sb_data_miss_impute.xy()
        dist_mat_miss_impute = metrics.pairwise_distances(x, y)
        the_kernel = kernel.KGauss(np.median(dist_mat_miss_impute))
        mmd_miss_impute = tst.QuadMMDTest(the_kernel, alpha=alpha)
    test_result = mmd_miss_impute.perform_test(sb_data_miss_impute)
    if test_result['h0_rejected']:
        mmd_result[0] = 1
    
    sb_data_full = TSTData(train_full, test_full)
    if mmd_full is None:
        x,y = sb_data_full.xy()
        dist_mat_full = metrics.pairwise_distances(x, y)
        the_kernel = kernel.KGauss(np.median(dist_mat_full))
        mmd_full = tst.QuadMMDTest(the_kernel, alpha=alpha)
    test_result = mmd_full.perform_test(sb_data_full)
    if test_result['h0_rejected']:
        mmd_result[1] = 1
        
    return mmd_result, mmd_miss_impute, mmd_full

def perform_me_test(train_miss_impute, test_miss_impute, train_full, test_full, alpha, 
                     test_locs_miss=None, gwidth_miss=None, test_locs_full=None, gwidth_full=None):
    me_result = np.zeros(2)
    
    op = {
    'n_test_locs': 5, # number of test locations to optimize
    'max_iter': 200, # maximum number of gradient ascent iterations
    'locs_step_size': 1.0, # step size for the test locations (features)
    'gwidth_step_size': 0.1, # step size for the Gaussian width
    'tol_fun': 1e-4, # stop if the objective does not increase more than this.
    'seed': 0  # random seed
    }
    
    sb_data_miss_impute = TSTData(train_miss_impute, test_miss_impute)
    train_miss_impute_sb, dumy = sb_data_miss_impute.split_tr_te(tr_proportion=0.1, seed=1)
    test_miss_impute_sb, dumy = sb_data_miss_impute.split_tr_te(tr_proportion=1, seed=1)
    
    if test_locs_miss is None:
        test_locs_miss, gwidth_miss, info = tst.MeanEmbeddingTest.optimize_locs_width(train_miss_impute_sb, alpha, **op)
    met_opt = tst.MeanEmbeddingTest(test_locs_miss, gwidth_miss, alpha)
    test_result = met_opt.perform_test(test_miss_impute_sb)
    if test_result['h0_rejected']:
        me_result[0] = 1
    
    sb_data_full = TSTData(train_full, test_full)
    train_full_sb, dumy = sb_data_full.split_tr_te(tr_proportion=0.1, seed=1)
    test_full_sb, dumy = sb_data_full.split_tr_te(tr_proportion=1, seed=1)
    #dumy, test_full_sb = sb_data_full.split_tr_te(tr_proportion=0, seed=1)
    
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
        print('pearson_dist_free', K, m1, m2, threshold)
    
    hp1, _ = Qtree_Htest_miss_impute.reject_null_hypothesis(test_miss_impute, alpha)
    if hp1:
        QTree_result[0] = 1
        
    if Qtree_Htest_full is None:
        qtree_full = qt.QuantTree(K)
        qtree_full.build_histogram(train_full, True)
        Qtree_Htest_full = qt.ChangeDetectionTest(qtree_full, m2, qt.pearson_statistic)
        threshold = qt.ChangeDetectionTest.get_precomputed_quanttree_threshold('pearson', alpha, K, m1, m2)
        if threshold is None:
            threshold = Qtree_Htest_full.estimate_quanttree_threshold(alpha, B)
        Qtree_Htest_full.set_threshold(alpha, threshold)
        print('pearson_dist_free', K, m1, m2, threshold)
    
    hp2, _ = Qtree_Htest_full.reject_null_hypothesis(test_full, alpha)
    if hp2:
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


def perform_mfkmean_chi2_test(train_miss, test_miss, train_full, test_full, alpha, mfkchi2_miss=None, mfkchi2_full=None, apply_fuzzy='Rectangle', top_k=3):
    
    m = train_miss.shape[0]
    mfkmean_chi2_result = np.zeros(2)
    
    if mfkchi2_miss is None:
        mfkchi2_miss = mf_distance_kmeans_chi2_test.MFkMeansChi2(int(m/50), apply_fuzzy, top_k=top_k)
        mfkchi2_miss.buildMFkMeans(train_miss)
        
    if mfkchi2_full is None:
        mfkchi2_full = mf_distance_kmeans_chi2_test.MFkMeansChi2(int(m/50), apply_fuzzy, top_k=top_k)
        mfkchi2_full.buildMFkMeans(train_full)
    
    mfkmean_chi2_result[0] = mfkchi2_miss.drift_detection(test_miss, alpha)
    mfkmean_chi2_result[1] = mfkchi2_full.drift_detection(test_full, alpha)
    
    return mfkmean_chi2_result, mfkchi2_miss, mfkchi2_full

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

def evaluate_exp4(main_folder, mv_config, r_seed = 0, num_file = 500):
       
    train_with_nan, train_full, w0_with_nan_list, w0_list, w1_with_nan_list, w1_list = dh.load_realdata(r_seed, mv_config, num_file, main_folder)
    
    imp = IterativeImputer(max_iter=10, random_state=0)
    train_impu = imp.fit_transform(train_with_nan)
        
    alpha = 0.05
    
    result_MWW = np.zeros([2, 2])
    result_QTree = np.zeros([2, 2])
    result_kchi2 = np.zeros([2, 2])
    result_Gau = np.zeros([2, 2])
    result_Tri = np.zeros([2, 2])
    result_ME = np.zeros([2, 2])
    result_MMD = np.zeros([2, 2])
    
    Qtree_Htest_impu = None
    Qtree_Htest_full = None
    
    kchi2_impu = None
    kchi2_full = None
    
    mfkchi2_miss_gau = None
    mfkchi2_full_gau = None
    mfkchi2_miss_tri = None
    mfkchi2_full_tri = None
    
    me_ml = None
    me_mg = None
    me_fl = None
    me_fg = None

    mmd_impu = None
    mmd_full = None
    
    for i in range(num_file):
        
        imp = IterativeImputer(max_iter=10, random_state=0)
        w0_miss = w0_with_nan_list[i][0]
        w0_impu = imp.fit_transform(w0_miss)
        w0_full = w0_list[i]
        
        imp = IterativeImputer(max_iter=10, random_state=0)
        w1_miss = w1_with_nan_list[i][0]
        w1_impu = imp.fit_transform(w1_miss)
        w1_full = w1_list[i]
        
        alg_r_seed = 1
        print('MWW')
        np.random.seed(alg_r_seed)
        # ============================================================================================================== #
        w0_result = perform_mww_test(train_impu, w0_impu, train_full, w0_full, alpha)
        result_MWW[0] = result_MWW[0] + w0_result
        w1_result = perform_mww_test(train_impu, w1_impu, train_full, w1_full, alpha)
        result_MWW[1] = result_MWW[1] + w1_result
        
        print('Qtree')
        np.random.seed(alg_r_seed)
        # ============================================================================================================== #
        w0_result, Qtree_Htest_impu, Qtree_Htest_full = perform_QTree_test(train_impu, w0_impu, train_full, w0_full, alpha, 
                                                                           Qtree_Htest_impu, Qtree_Htest_full)
        result_QTree[0] = result_QTree[0] + w0_result
        w1_result, Qtree_Htest_impu, Qtree_Htest_full = perform_QTree_test(train_impu, w1_impu, train_full, w1_full, alpha, 
                                                                           Qtree_Htest_impu, Qtree_Htest_full)
        result_QTree[1] = result_QTree[1] + w1_result
#        
        print('kchi2')
        np.random.seed(alg_r_seed)
        # ============================================================================================================== #
        w0_result, kchi2_impu, kchi2_full = perform_kmean_chi2_test(train_impu, w0_impu, train_full, w0_full, alpha, 
                                                                      kchi2_impu, kchi2_full)
        result_kchi2[0] = result_kchi2[0] + w0_result
        w1_result, kchi2_impu, kchi2_full = perform_kmean_chi2_test(train_impu, w1_impu, train_full, w1_full, alpha, 
                                                                      kchi2_impu, kchi2_full)
        result_kchi2[1] = result_kchi2[1] + w1_result
        
        print('Gau')
        np.random.seed(alg_r_seed)
        # ============================================================================================================== #
        w0_result, mfkchi2_miss_gau, mfkchi2_full_gau = perform_mfkmean_chi2_test(train_with_nan, w0_miss, train_full, w0_full, alpha,
                                                                            mfkchi2_miss_gau, mfkchi2_full_gau, apply_fuzzy='Gaussion', top_k=2)
        result_Gau[0] = result_Gau[0] + w0_result
        w1_result, mfkchi2_miss_gau, mfkchi2_full_gau = perform_mfkmean_chi2_test(train_with_nan, w1_miss, train_full, w1_full, alpha, 
                                                                            mfkchi2_miss_gau, mfkchi2_full_gau, apply_fuzzy='Gaussion', top_k=2)
        result_Gau[1] = result_Gau[1] + w1_result
        
        print('Tri')
        np.random.seed(alg_r_seed)
        # ============================================================================================================== #
        w0_result, mfkchi2_miss_tri, mfkchi2_full_tri = perform_mfkmean_chi2_test(train_with_nan, w0_miss, train_full, w0_full, alpha,
                                                                            mfkchi2_miss_tri, mfkchi2_full_tri, apply_fuzzy='Triangle', top_k=2)
        result_Tri[0] = result_Tri[0] + w0_result
        w1_result, mfkchi2_miss_tri, mfkchi2_full_tri = perform_mfkmean_chi2_test(train_with_nan, w1_miss, train_full, w1_full, alpha, 
                                                                            mfkchi2_miss_tri, mfkchi2_full_tri, apply_fuzzy='Triangle', top_k=2)
        result_Tri[1] = result_Tri[1] + w1_result

        print('ME')
        np.random.seed(alg_r_seed)
        # ============================================================================================================== #
        w0_result, me_ml, me_mg, me_fl, me_fg = perform_me_test(train_impu, w0_impu, train_full, w0_full, alpha,
                                                                            me_ml, me_mg, me_fl, me_fg)
        result_ME[0] = result_ME[0] + w0_result
        w1_result, me_ml, me_mg, me_fl, me_fg = perform_me_test(train_impu, w1_impu, train_full, w1_full, alpha, 
                                                                            me_ml, me_mg, me_fl, me_fg)
        result_ME[1] = result_ME[1] + w1_result
#        
        print('MMD')
        np.random.seed(alg_r_seed)
        # ============================================================================================================== #
        w0_result, mmd_impu, mmd_full = perform_mmd_test(train_impu, w0_impu, train_full, w0_full, alpha, mmd_impu, mmd_full)
        result_MMD[0] = result_MMD[0] + w0_result
        w1_result, mmd_impu, mmd_full = perform_mmd_test(train_impu, w1_impu, train_full, w1_full, alpha, mmd_impu, mmd_full)
        result_MMD[1] = result_MMD[1] + w1_result
#        
    return result_MWW, result_QTree, result_kchi2, result_Gau, result_Tri, result_ME, result_MMD
        
def evaluate_HiggBack():
    pass
    
if __name__ == "__main__":
    
    main_folder1 = 'Data/Exp4_ReaDriftDetection/1_Higgs/Data/back_higg/'
    main_folder2 = 'Data/Exp4_ReaDriftDetection/1_Higgs/Data/higg_back/'
    main_folder3 = 'Data/Exp4_ReaDriftDetection/2_Arabic_Digit/Data/mix1_mix2/'
    main_folder4 = 'Data/Exp4_ReaDriftDetection/2_Arabic_Digit/Data/mix2_mix1/'
    main_folder5 = 'Data/Exp4_ReaDriftDetection/3_Insects/Data/mix1_mix2/'
    main_folder6 = 'Data/Exp4_ReaDriftDetection/3_Insects/Data/mix2_mix1/'
    
    r_seed = 0
    num_file = 50
    
    mv_config = {}
    for i in range(4):
        mv_config[i] = 0.2
    result1 = evaluate_exp4(main_folder1, mv_config, r_seed, num_file)
    result2 = evaluate_exp4(main_folder2, mv_config, r_seed, num_file)
    
    
    mv_config = {}
    drift_feats = [3, 4, 16, 25]
    for i in drift_feats:
        mv_config[i] = 0.2
    result3 = evaluate_exp4(main_folder3, mv_config, r_seed, num_file)
    
    mv_config = {}
    drift_feats = [1, 3, 4, 12, 16, 19]
    for i in drift_feats:
        mv_config[i] = 0.2
    result4 = evaluate_exp4(main_folder4, mv_config, r_seed, num_file)
    
    
    
    mv_config = {}
    drift_feats = [2, 5, 13, 18]
    for i in drift_feats:
        mv_config[i] = 0.2
    result5 = evaluate_exp4(main_folder5, mv_config, r_seed, num_file)
    
    mv_config = {}
    drift_feats = [1, 3, 4, 16, 32, 47]
    for i in drift_feats:
        mv_config[i] = 0.2
    result6 = evaluate_exp4(main_folder6, mv_config, r_seed, num_file)
    
    
    r1_all = np.vstack(result1)
    r2_all = np.vstack(result2)
    r3_all = np.vstack(result3)
    r4_all = np.vstack(result4)
    r5_all = np.vstack(result5)
    r6_all = np.vstack(result6)
    
    final_r = np.hstack([ r1_all, r2_all, r3_all, r4_all, r5_all, r6_all])
    
    the_r_seed = 1
    np.savetxt('final_r_seed_'+the_r_seed+'.csv', final_r, delimiter=',')
    
    