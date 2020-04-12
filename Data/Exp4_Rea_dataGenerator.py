# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 16:40:14 2019

@author: Anjin Liu (anjin.liu@uts.edu.au)

Experiment data set for paper submitted to TCYB2019,
Title: Concept Drift Detection via Equal Intensity k-means Space Partitioning
"""
import scipy.io as sio
import numpy as np
import os

def generate_Higgs(r_seed, file_back, file_higg, datafolder_path, N_train, N_test, num_test):
    print('Generating Higgs-Back and Back-Higgs')
    np.random.seed(r_seed)
    
    data_back = sio.loadmat(file_back)['data0']
    data_higg = sio.loadmat(file_higg)['data1']
    
    ################
    # data shuffle #
    ################
    
    np_back = np.asarray(data_back)
    np_higg = np.asarray(data_higg)

    np.random.shuffle(np_back)
    np.random.shuffle(np_higg)
    
    total_samples = N_train + N_test*num_test
    
    np_back = np_back[:total_samples,:-1]
    np_higg = np_higg[:total_samples,:-1]
    
    np_back_data_train = np_back[:N_train]
    np_higg_data_train = np_higg[:N_train]
    
    output_data_folder_bh = datafolder_path + "back_higg"
    os.makedirs(output_data_folder_bh, exist_ok=True)
    np.savetxt(output_data_folder_bh+'/data_train.csv', np_back_data_train, delimiter=',', fmt='%s')
    
    output_data_folder_hb = datafolder_path + "higg_back"
    os.makedirs(output_data_folder_hb, exist_ok=True)
    np.savetxt(output_data_folder_hb+'/data_train.csv', np_higg_data_train, delimiter=',', fmt='%s')
    
    for t in range(N_train, total_samples, N_test):
        
        b = np_back[t:t+N_test]
        h = np_higg[t:t+N_test]
        
        np.savetxt(output_data_folder_bh+'/data_w0_'+str(int((t-N_train)/N_test))+'.csv', b, delimiter=',', fmt='%s')
        np.savetxt(output_data_folder_bh+'/data_w1_'+str(int((t-N_train)/N_test))+'.csv', h, delimiter=',', fmt='%s')
        np.savetxt(output_data_folder_hb+'/data_w0_'+str(int((t-N_train)/N_test))+'.csv', h, delimiter=',', fmt='%s')
        np.savetxt(output_data_folder_hb+'/data_w1_'+str(int((t-N_train)/N_test))+'.csv', b, delimiter=',', fmt='%s')

def generate_ArabicDigit(r_seed, datafolder_path, datafile_name, N_train, N_test, num_test):
    print('Generating Arabic A-B and B-A')
    np.random.seed(r_seed)
    
    mixture_1 = ['m0','m1','m2','m3','m4','f5','f6','f7','f8','f9']
    mixture_2 = ['m0','m1','m2','m3','m4','f5','f6','f7','f8','m9']
    
    file = open(datafile_name, "r")    
    
    data = {}
    for i in range(len(mixture_1)):
        data['m'+str(i)] = []
    for i in range(len(mixture_1)):
        data['f'+str(i)] = []
    
    
    line_counter = 0
    for line in file:
        items = line.split(',')
        if 0 < line_counter:
            if items[-2]=='male':
                data['m'+items[-1].replace('\n', '')].append(','.join(items[:-2]))
            else:
                data['f'+items[-1].replace('\n', '')].append(','.join(items[:-2]))
        line_counter += 1
    file.close()
    
    ################
    # data shuffle #
    ################
    
    mixture_1_data = []
    mixture_2_data = []
    for item in mixture_1:
        mixture_1_data += data[item]
    for item in mixture_2:
        mixture_2_data += data[item]   
    
    mixture_1_data = np.array(mixture_1_data)
    mixture_2_data = np.array(mixture_2_data)
    mixture_1_num = len(mixture_1_data)
    mixture_2_num = len(mixture_2_data)
    
    idx_array_1 = np.arange(mixture_1_num)
    idx_array_2 = np.arange(mixture_2_num)
    
    np.random.shuffle(idx_array_1)
    np.random.shuffle(idx_array_2)
    mixture_1_train_idx = idx_array_1[:N_train]
    mixture_2_train_idx = idx_array_2[:N_train]
    idx_array_1 = idx_array_1[N_train:]
    idx_array_2 = idx_array_2[N_train:]
    
    mixture_1_data_train = mixture_1_data[mixture_1_train_idx]
    mixture_2_data_train = mixture_2_data[mixture_2_train_idx]
    
    output_data_folder_mf = datafolder_path + "mix1_mix2"
    os.makedirs(output_data_folder_mf, exist_ok=True)
    np.savetxt(output_data_folder_mf+'/data_train.csv', mixture_1_data_train, fmt='%s')
    
    output_data_folder_fm = datafolder_path + "mix2_mix1"
    os.makedirs(output_data_folder_fm, exist_ok=True)
    np.savetxt(output_data_folder_fm+'/data_train.csv', mixture_2_data_train, fmt='%s')
    
    for t in range(num_test):
        
        np.random.shuffle(idx_array_1)
        np.random.shuffle(idx_array_2)
        m1m2_W0_idx = idx_array_1[:N_test]
        m1m2_W1_idx = idx_array_2[:N_test]
        
        np.random.shuffle(idx_array_1)
        np.random.shuffle(idx_array_2)
        m2m1_W0_idx = idx_array_2[:N_test]
        m2m1_W1_idx = idx_array_1[:N_test]

        W0_m1m2 = mixture_1_data[m1m2_W0_idx]
        W1_m1m2 = mixture_2_data[m1m2_W1_idx]
        W0_m2m1 = mixture_2_data[m2m1_W0_idx]
        W1_m2m1 = mixture_1_data[m2m1_W1_idx]
        
        np.savetxt(output_data_folder_mf+'/data_w0_'+str(t)+'.csv', W0_m1m2, fmt='%s')
        np.savetxt(output_data_folder_mf+'/data_w1_'+str(t)+'.csv', W1_m1m2, fmt='%s')
        np.savetxt(output_data_folder_fm+'/data_w0_'+str(t)+'.csv', W0_m2m1, fmt='%s')
        np.savetxt(output_data_folder_fm+'/data_w1_'+str(t)+'.csv', W1_m2m1, fmt='%s')
        
        
def generate_Insects(r_seed, datafolder_path, datafile_name, num_test):
    
    print('Generating Insect')
    np.random.seed(r_seed)
    
    file = open(datafile_name, "r")    
    insect_type = ['flies', 'aedes', 'tarsalis', 'quinx', 'fruit']
    timer = 2
    mix1_config = {'flies':100*timer, 'aedes':100*timer, 'tarsalis':100*timer, 'quinx':100*timer, 'fruit':100*timer}
    mix2_config = {'flies':80*timer, 'aedes':90*timer, 'tarsalis':100*timer, 'quinx':110*timer, 'fruit':120*timer}
    data = {}    
    
    for line in file:
        line = line.replace('\n','')
        items = line.split(',')
        if items[-1] in data:
            data[items[-1]].append(','.join(items[:-1]))
        else:
            data[items[-1]] = [','.join(items[:-1])]
    file.close()
        
    ################
    # data shuffle #
    ################
    
    mix12_train_data = np.array([])
    mix21_train_data = np.array([])
    
    mix12_test_data = {}
    mix21_test_data = {}
    
    for insect_t in insect_type:
        np.random.shuffle(data[insect_t])
        mix12_train_data = np.append(mix12_train_data, data[insect_t][:mix1_config[insect_t]])
        mix21_train_data = np.append(mix21_train_data, data[insect_t][:mix2_config[insect_t]])
        
        mix12_test_data[insect_t] = data[insect_t][mix1_config[insect_t]:]
        mix21_test_data[insect_t] = data[insect_t][mix2_config[insect_t]:]
    
    output_data_folder_m12 = datafolder_path + "mix1_mix2"
    output_data_folder_m21 = datafolder_path + "mix2_mix1"
    os.makedirs(output_data_folder_m12, exist_ok=True)
    os.makedirs(output_data_folder_m21, exist_ok=True)
    
    np.savetxt(output_data_folder_m12+'/data_train.csv', mix12_train_data, fmt='%s')
    np.savetxt(output_data_folder_m21+'/data_train.csv', mix21_train_data, fmt='%s')
    
    for t in range(num_test):
        w0 = np.array([])
        for insect_t in insect_type:
            np.random.shuffle(mix12_test_data[insect_t])
            w0 = np.append(w0, mix12_test_data[insect_t][:mix1_config[insect_t]])
        np.random.shuffle(w0)
        
        w1 = np.array([])
        for insect_t in insect_type:
            np.random.shuffle(mix12_test_data[insect_t])
            w1 = np.append(w1, mix12_test_data[insect_t][:mix2_config[insect_t]])
        np.random.shuffle(w1)
        
        np.savetxt(output_data_folder_m12+'/data_w0_'+str(t)+'.csv', w0, fmt='%s')
        np.savetxt(output_data_folder_m12+'/data_w1_'+str(t)+'.csv', w1, fmt='%s')
        
        
        w0 = np.array([])
        for insect_t in insect_type:
            np.random.shuffle(mix21_test_data[insect_t])
            w0 = np.append(w0, mix21_test_data[insect_t][:mix2_config[insect_t]])
        np.random.shuffle(w0)
        
        w1 = np.array([])
        for insect_t in insect_type:
            np.random.shuffle(mix21_test_data[insect_t])
            w1 = np.append(w1, mix21_test_data[insect_t][:mix1_config[insect_t]])
        np.random.shuffle(w1)
        
        np.savetxt(output_data_folder_m21+'/data_w0_'+str(t)+'.csv', w0, fmt='%s')
        np.savetxt(output_data_folder_m21+'/data_w1_'+str(t)+'.csv', w1, fmt='%s')
        
        
if __name__ == "__main__":
    
    num_test = 500
    r_seed = 2
    generate_Higgs(r_seed, "OriginalDataFiles/1_Higgs/data0", "OriginalDataFiles/1_Higgs/data1", "Exp4_ReaDriftDetection/1_Higgs/Data/", 1000, 1000, num_test)
    generate_ArabicDigit(r_seed, "Exp4_ReaDriftDetection/2_Arabic_Digit/Data/", "OriginalDataFiles/2_Arabic_Digit/ArabicDigit_Shuffled_With_Sex.csv", 1000, 1000, num_test)
    generate_Insects(r_seed, "Exp4_ReaDriftDetection/3_Insects/Data/", "OriginalDataFiles/3_Insects/Insects.data", num_test)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
