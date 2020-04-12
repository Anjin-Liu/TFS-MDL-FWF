# -*- coding: utf-8 -*-
"""
Created on Fri May 31 17:23:04 2019

@author: Anjin Liu
version issue:
    "Data cannot be evenly distributed into partitions m%k!=0"
"""

from sklearn.cluster import KMeans
import numpy as np
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
from mf_distance import dist_estimator as de
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics.pairwise import euclidean_distances
from scipy.stats import norm
from scipy.special import softmax

class MFkMeans():
    
    def __init__(self, k, top_k=3, std_est_kfold=2):
        self.k = k
        self.dist_est = None
        self.cluster_centers_ = None
        self.dist_est_error_mean = 0
        self.dist_est_error_std = 0
        self.std_est_kfold = std_est_kfold
        self.top_k = top_k
        
    def fit(self, data_train):
        
        X_noNan_idx = np.where(~np.isnan(data_train).any(axis=1))[0]
        X_noNan = data_train[X_noNan_idx]
        kmeans = KMeans(n_clusters=self.k, random_state=0).fit(X_noNan)
        self.cluster_centers_ = kmeans.cluster_centers_
        #========================#
        # estiamte missing ratio #
        #========================#
        nan_ratio = (np.isnan(data_train)*1).sum(axis=0)
        nan_ratio = nan_ratio/data_train.shape[0]
        
        if nan_ratio.sum() > 0:
            mv_config = {}
            for i in range(nan_ratio.shape[0]):
                mv_config[i] = nan_ratio[i]
            
            train_X, train_Y = de.extract_feats_masking(X_noNan, mv_config)
            
            kf = KFold(n_splits=self.std_est_kfold)
            diff = np.zeros(train_X.shape[0])
            counter = 0
            for train_index, test_index in kf.split(train_X):
                kf_X_train, kf_X_test = train_X[train_index], train_X[test_index]
                kf_Y_train, kf_Y_test = train_Y[train_index], train_Y[test_index]
                kf_gbr = GradientBoostingRegressor(loss='huber', n_estimators=500, learning_rate=0.05, max_depth=4, random_state=0)
                kf_gbr.fit(kf_X_train, kf_Y_train)
                kf_Y_pred = kf_gbr.predict(kf_X_test)
                diff[test_index] = kf_Y_test - kf_Y_pred
                counter += 1
            self.dist_est_error_mean = np.mean(diff)
            
            self.dist_est_error_std = np.std(diff)
            #self.dist_est_error_std = np.mean(np.abs(diff))
            print(np.std(diff), self.dist_est_error_std, np.mean(train_Y))
            self.dist_est = GradientBoostingRegressor(loss='huber', n_estimators=500, learning_rate=0.05, max_depth=4, random_state=0)
            self.dist_est.fit(train_X, train_Y)
    
    def predict_crisp(self, data_test):
        
        if self.dist_est is not None:
            
            X_noNan_idx = np.where(~np.isnan(data_test).any(axis=1))[0]
            X_Nan_idx = np.where(np.isnan(data_test).any(axis=1))[0]
            X_noNan = data_test[X_noNan_idx]
            X_Nan = data_test[X_Nan_idx]
            C_idx = np.zeros(data_test.shape[0])
            
            m = X_Nan.shape[0]
            data_test_masked = de.extract_feats_transform(self.cluster_centers_, X_Nan)
            est_dist_flatten = self.dist_est.predict(data_test_masked)
            est_dist_mat_pre = est_dist_flatten.reshape([self.k, m])
            C_idx_Nan = np.argmin(est_dist_mat_pre, axis=0)
            
            est_dist_mat = euclidean_distances(self.cluster_centers_, X_noNan)
            C_idx_noNan = np.argmin(est_dist_mat, axis=0)
            
            C_idx[X_Nan_idx] = C_idx_Nan
            C_idx[X_noNan_idx] = C_idx_noNan
            
        else:
            
            C_idx = euclidean_distances(self.cluster_centers_, data_test)
            C_idx = np.argmin(C_idx, axis=0)
            
        return C_idx
    
    def predict_fuzzy(self, data_test, membership_function='Rectangle'):
        
        if self.dist_est is not None:
            
            X_noNan_idx = np.where(~np.isnan(data_test).any(axis=1))[0]
            X_Nan_idx = np.where(np.isnan(data_test).any(axis=1))[0]
            X_noNan = data_test[X_noNan_idx]
            X_Nan = data_test[X_Nan_idx]
            C_idx_fuzzy = np.zeros([self.k, data_test.shape[0]])
            
            m = X_Nan.shape[0]
            data_test_masked = de.extract_feats_transform(self.cluster_centers_, X_Nan)
            est_dist_flatten = self.dist_est.predict(data_test_masked)
            est_dist_mat_pre = est_dist_flatten.reshape([self.k, m])
            
            if membership_function == 'Gaussion':
                fuzzy_membership = self.resolve_fuzzy_samples_gau(est_dist_mat_pre)
            elif membership_function == 'Triangle':
                fuzzy_membership = self.resolve_fuzzy_samples_tri(est_dist_mat_pre)
            elif membership_function == 'Rectangle':
                fuzzy_membership = self.resolve_fuzzy_samples_rec(est_dist_mat_pre)
            else:
                raise ValueError("Given apply_fuzzy not in {Gaussion, Triangle, Rectangle, Crisp}")
                
            C_idx_fuzzy[:, X_Nan_idx] = fuzzy_membership
            
            est_dist_mat = euclidean_distances(self.cluster_centers_, X_noNan)
            C_idx_noNan = np.argmin(est_dist_mat, axis=0)
            m = np.arange(C_idx_noNan.shape[0])
            one_membership_idx = np.vstack([C_idx_noNan, m]).T
            m = C_idx_noNan.shape[0]
            one_membership = np.zeros([self.k, m])
            one_membership[one_membership_idx[:,0], one_membership_idx[:,1]] = 1
            C_idx_fuzzy[:, X_noNan_idx] = one_membership
            
        else:
            
            C_idx = euclidean_distances(self.cluster_centers_, data_test)
            C_idx = np.argmin(C_idx, axis=0)
            m = np.arange(C_idx.shape[0])
            one_membership_idx = np.vstack([C_idx, m]).T
            m = C_idx.shape[0]
            one_membership = np.zeros([self.k, m])
            one_membership[one_membership_idx[:,0], one_membership_idx[:,1]] = 1
            C_idx_fuzzy = one_membership
            
        return C_idx_fuzzy
    
    def resolve_fuzzy_samples_gau(self, est_dist_mat_pre):
        
        min_dist_vec = np.min(est_dist_mat_pre, axis=0)
        min_dist_mat = np.dot(np.ones([self.k, 1]), np.array([min_dist_vec]))
        gau_intersection_point = (est_dist_mat_pre - min_dist_mat)/2 + min_dist_mat

        std_mat = np.ones(min_dist_mat.shape)* self.dist_est_error_std
        intersection_area = (1.-norm.cdf(gau_intersection_point, min_dist_mat, std_mat))*2
        
        top_k_cluster_threshold_vec = np.sort(intersection_area, axis=0)[-self.top_k,:]
        top_k_cluster_threshold_mat = np.dot(np.ones([self.k, 1]), np.array([top_k_cluster_threshold_vec]))
        
        intersection_area = np.where(intersection_area >= top_k_cluster_threshold_mat, intersection_area, 0)
        
        fuzzy_membership = intersection_area/np.dot(np.ones([self.k, 1]), np.array([np.sum(intersection_area, axis=0)]))
        #fuzzy_membership = fuzzy_membership/np.dot(np.ones([self.k, 1]), np.array([np.sum(fuzzy_membership, axis=0)]))
#        fuzzy_membership = softmax(intersection_area, axis=0)
        
        return fuzzy_membership
    
    def resolve_fuzzy_samples_tri(self, est_dist_mat_pre):
        
        est_dist_mat_pre_min = est_dist_mat_pre - self.dist_est_error_std
        
        min_dist_upperbound = np.min(est_dist_mat_pre, axis=0) + self.dist_est_error_std
        min_dist_upperbound = np.dot(np.ones([self.k, 1]), np.array([min_dist_upperbound]))
        fuzzy_membership = min_dist_upperbound - est_dist_mat_pre_min
        
        top_k_cluster_threshold_vec = np.sort(fuzzy_membership, axis=0)[-self.top_k,:]
        top_k_cluster_threshold_mat = np.dot(np.ones([self.k, 1]), np.array([top_k_cluster_threshold_vec]))
        
        fuzzy_membership = np.where(fuzzy_membership >= top_k_cluster_threshold_mat, fuzzy_membership, 0)
        fuzzy_membership = fuzzy_membership / (self.dist_est_error_std*2)
        fuzzy_membership = fuzzy_membership**2
        fuzzy_membership = fuzzy_membership/np.dot(np.ones([self.k, 1]), np.array([np.sum(fuzzy_membership, axis=0)]))
        
#        fuzzy_membership = softmax(fuzzy_membership, axis=0)
        return fuzzy_membership
    
    def resolve_fuzzy_samples_rec(self, est_dist_mat_pre):
        est_dist_mat_pre_min = est_dist_mat_pre - self.dist_est_error_std
        min_dist_upperbound = np.min(est_dist_mat_pre, axis=0) + self.dist_est_error_std
        min_dist_upperbound = np.dot(np.ones([self.k, 1]), np.array([min_dist_upperbound]))
        fuzzy_membership = min_dist_upperbound - est_dist_mat_pre_min
        
        top_k_cluster_threshold_vec = np.sort(fuzzy_membership, axis=0)[-self.top_k,:]
        top_k_cluster_threshold_mat = np.dot(np.ones([self.k, 1]), np.array([top_k_cluster_threshold_vec]))
        
        fuzzy_membership = np.where(fuzzy_membership >= top_k_cluster_threshold_mat, fuzzy_membership, 0)
        fuzzy_membership = fuzzy_membership / (self.dist_est_error_std*2)
        fuzzy_membership = fuzzy_membership/np.dot(np.ones([self.k, 1]), np.array([np.sum(fuzzy_membership, axis=0)]))
        
        return fuzzy_membership
    
class MFkMeansChi2():
    
    def __init__(self, k, apply_fuzzy='Crisp', top_k=3):
        
        self.mfkmeans = None
        self.k = k
        self.lambdas = None
        self.dist_est = None
        self.apply_fuzzy = apply_fuzzy
        self.top_k = top_k
    
    def buildMFkMeans(self, data_train, output_path=None, plot_partition=False):
        
        k = self.k
        print(1)
        self.mfkmeans = MFkMeans(k, top_k=self.top_k)
        print(2)
        data_train_copy = np.array(data_train)
        np.random.shuffle(data_train_copy)
        data_train_sub = data_train[:3000]
        print(3)
        self.mfkmeans.fit(data_train_sub)
        print(4)
        #if output_path is not None:
        #    self.output_partition_result(X_noNan, C_idx, output_path)
    
        #===========#
        # plot test #
        #===========#
        '''
        if plot_partition:
            for i in range(self.k):
                idx = np.where(C_idx == i)
                plt.scatter(data_train[idx[0], 0], data_train[idx[0], 1])
            plt.show()
        '''
        if self.apply_fuzzy=='Gaussion':
            C_idx = self.mfkmeans.predict_fuzzy(data_train, self.apply_fuzzy)
            self.lambdas = C_idx.sum(axis=1)
            
        elif self.apply_fuzzy=='Triangle':
            C_idx = self.mfkmeans.predict_fuzzy(data_train, self.apply_fuzzy)
            self.lambdas = C_idx.sum(axis=1)
            
        elif self.apply_fuzzy=='Rectangle':
            C_idx = self.mfkmeans.predict_fuzzy(data_train, self.apply_fuzzy)
            self.lambdas = C_idx.sum(axis=1)
            
        elif self.apply_fuzzy=='Crisp':
            C_idx = self.mfkmeans.predict_crisp(data_train)
            self.lambdas = np.zeros(k)
            k_list, unique_count = np.unique(C_idx, return_counts=True)
            self.lambdas[k_list.astype(int)] = unique_count
        else:
            raise ValueError("Given apply_fuzzy not in {Gaussion, Triangle, Rectangle, Crisp}")
        
    def drift_detection(self, data_test, alpha):
        
        if self.apply_fuzzy=='Gaussion':
            C_idx = self.mfkmeans.predict_fuzzy(data_test, self.apply_fuzzy)
            observations = C_idx.sum(axis=1)
            
        elif self.apply_fuzzy=='Triangle':
            C_idx = self.mfkmeans.predict_fuzzy(data_test, self.apply_fuzzy)
            observations = C_idx.sum(axis=1)
            
        elif self.apply_fuzzy=='Rectangle':
            C_idx = self.mfkmeans.predict_fuzzy(data_test, self.apply_fuzzy)
            observations = C_idx.sum(axis=1)
            
        elif self.apply_fuzzy=='Crisp':
            C_idx = self.mfkmeans.predict_crisp(data_test)
            observations = np.zeros(self.k)
            k_list, unique_count = np.unique(C_idx, return_counts=True)
            observations[k_list.astype(int)] = unique_count
        else:
            raise ValueError("Given apply_fuzzy not in {Gaussion, Triangle, Rectangle, Crisp}")
            
        h = 0
        contingency_table = np.array([self.lambdas, observations])
        chi2, p, dof, ex = chi2_contingency(contingency_table)
        if p < alpha:
            h = 1
        return h
    
    '''
    def output_partition_result(self, X, y, output_path):
        
        output_data = np.hstack([X, y.reshape(-1,1)])
        np.savetxt(output_path, output_data, delimiter=",")
    '''   
        
        
        
        
        