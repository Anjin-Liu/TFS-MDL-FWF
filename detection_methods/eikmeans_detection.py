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
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import pairwise_kernels
from pyclustering.cluster.kmedoids import kmedoids
from copy import deepcopy
from sklearn.neighbors import NearestNeighbors

class EIkMedoids():
    
    def __init__(self, k, lambdas=None, beta=1, current_k=0, fine_tune=None, metric_type='euc'):
        
        self.k = k
        self.lambdas = lambdas
        self.beta = beta
        self.current_k = current_k
        self.fine_tune = fine_tune
        self.theta = np.arange(0.0, 1, 0.05)
        self.metric = metric_type
        self.train_knn_hist = None
        self.ini_data = None
        self.norm_max = -1
    
    def get_copy(self):
        new_copy = EIkMedoids(self.k, self.lambdas, self.beta, self.current_k, deepcopy(self.fine_tune))
        return new_copy
    
    def self_defined_knn_classificaition(self, data):
  
        data_knn_hist, data_dist_mat = self.get_kNN_hist(data)
        
        dist_mat_norm = euclidean_distances(self.data_ini, data)
        dist_mat_norm = dist_mat_norm/self.norm_max
        #dist_mat_norm = dist_mat_norm/self.norm_max
        #dist_mat_norm = 1-pairwise_kernels(self.data_ini, data, metric='rbf')
        
        EI_dist_mat = self.get_EI_distance(data_dist_mat, data_knn_hist, dist_mat_norm, self.knn_hist)
        lab_idx = np.argmin(EI_dist_mat, axis=0)
        C_idx = self.data_lab[lab_idx]
        #k_list, unique_count = np.unique(C_idx, return_counts=True)
        #===========#
        # plot test #
        #===========#
#        for i in np.flip(k_list):
#            idx = np.where(C_idx == i)
#            plt.scatter(data[idx[0], 0], data[idx[0], 1])
#        plt.show()
        
        return C_idx
        
    def fill_lambda(self, data):
        
#        X = data
#        C_idx =  self.fine_tune.predict(X)
#        k_list, unique_count = np.unique(C_idx, return_counts=True)
#        
#        print(unique_count)
#        #===========#
#        # plot test #
#        #===========#
#        for i in np.flip(k_list):
#            idx = np.where(C_idx == i)
#            plt.scatter(data[idx[0], 0], data[idx[0], 1])
#        plt.show()
#        
#        k_list, unique_count = np.unique(C_idx, return_counts=True)
#        self.lambdas = np.zeros(k_list.shape[0])
#        self.lambdas[k_list.astype(int)] = unique_count
        
        C_idx = self.self_defined_knn_classificaition(data)
        k_list, unique_count = np.unique(C_idx, return_counts=True)
        self.lambdas = np.zeros(k_list.shape[0])
        self.lambdas[k_list.astype(int)] = unique_count
        
        
    def build_partition(self, data_train, test_size):
        
        m = data_train.shape[0]
        if m > 2000:
            data_ini = data_train[:2000]
        else:
            data_ini = data_train
            
        m_ini = data_ini.shape[0]
        min_5 = test_size/5
        min_50 = m/50
        min_num_p = int(np.min([min_5, min_50]))
        self.k = np.min([min_num_p, self.k])
        k = self.k
        unique_count= [0]
        C_idx = np.zeros(m_ini)
        
        k += 1 
        num_insts_part = np.max([int(m_ini/(k-1))*0.5, 50])
        train_knn_hist, train_dist_mat = self.get_kNN_hist(data_ini)
        EISim_mat = self.get_EI_distance(train_dist_mat, train_knn_hist)
        
        num_insts_part = np.max([int(m_ini/(k-1))*0.5, 50])
        while np.min(unique_count) < num_insts_part:
            k -= 1
            print(k)
            num_insts_part = np.max([int(m_ini/(k-1))*0.5, 50])
            # greedy ini
            initial_medoids = self.greed_compact_partition(data_ini, k)
            
            # create K-Medoids algorithm for processing distance matrix instead of points
            kmedoids_instance = kmedoids(EISim_mat, initial_medoids, data_type='distance_matrix', itermax=500)
            
            # run cluster analysis and obtain results
            kmedoids_instance.process()
            
            clusters = kmedoids_instance.get_clusters()
            medoids_index = kmedoids_instance.get_medoids()
            
            # convert cluster array to C_idx
            counter = 0
            for items in clusters:
                C_idx[items] = counter
                counter += 1
            
            clf = KNeighborsClassifier(n_neighbors=int(num_insts_part/2))
            clf.fit(X, y_pseudo)
            
            k_list, unique_count = np.unique(C_idx, return_counts=True)
            dr = unique_count/(m_ini/k)
            for _theta in self.theta:
                if (dr.shape[0]) < k:
                    break
                amplify_coe = np.exp((dr-1)*_theta)
                C_idx = self.amplify_cluster(EISim_mat, amplify_coe, medoids_index)
                k_list, unique_count = np.unique(C_idx, return_counts=True)
                dr = unique_count/int(m_ini/k)
                if np.min(unique_count) > num_insts_part:
                    break
        
        #===========#
        # fine tune #
        #===========#
        y_pseudo = C_idx
        X = data_ini
        clf = KNeighborsClassifier(n_neighbors=int(num_insts_part/2))
        clf.fit(X, y_pseudo) 
        self.fine_tune = clf
        self.fill_lambda(data_train)
        
    def drift_detection(self, data_test, alpha):
    
        lambdas = self.lambdas
        k = len(lambdas)
        observations = np.zeros(k)
        
        C_idx = self.self_defined_knn_classificaition(data_test)
        k_list, unique_count = np.unique(C_idx, return_counts=True)
        observations[k_list.astype(int)] = unique_count
        contingency_table = np.array([lambdas, observations])
        chi2, p, dof, ex = chi2_contingency(contingency_table)
        h = 0
        if p < alpha:
            h = 1
        return h

    def amplify_cluster(self, dist_mat, amplify_coe, medoids_index):
        
        m = dist_mat.shape[0]
        k = amplify_coe.shape[0]
        C_X_dist = dist_mat[medoids_index]
        amplify_coe_mat = np.repeat(amplify_coe, m, axis=0)
        amplify_coe_mat = amplify_coe_mat.reshape(k, m)
        C_X_dist_amplified = C_X_dist*amplify_coe_mat
        np.argmin(amplify_coe_mat, axis=0)
        C_idx = np.argmin(C_X_dist_amplified, axis=0)
        return C_idx
    
    def greed_compact_partition(self, data, k):
        
        m = data.shape[0]
        p_size = int(m/k)
        temp_data = np.array(data)
        C_idx = np.zeros(m) - 1
        idx_list = np.arange(m)
        for i in range(k-1):
            nbrs = NearestNeighbors(n_neighbors=p_size, algorithm='ball_tree').fit(temp_data)
            distances, indices = nbrs.kneighbors(temp_data)
            greed_idx = np.argsort(distances[:,-1])[-1]
            C_idx[idx_list[indices[greed_idx]]] = int(i)
            temp_data = np.delete(temp_data, indices[greed_idx], axis=0)
            idx_list = np.delete(idx_list, indices[greed_idx])

        C_idx[np.where(C_idx==-1)[0]] = int(k-1)
        initial_medoids = np.zeros(k) - 1
        for i in range(k):
            initial_medoids[i] = np.where(C_idx==i)[0][0]
        return initial_medoids.astype(int)
    
    

        