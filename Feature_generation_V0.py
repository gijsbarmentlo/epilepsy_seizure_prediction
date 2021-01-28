# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 12:03:08 2021

@author: gijsb
"""

"""

Description of the dataset

Intracranial EEG (iEEG) data clips are organized in folders containing training 
and testing data for each human patient. The training data is organized into 
ten minute EEG clips labeled "Preictal" for pre-seizure data segments, or 
"Interictal" for non-seizure data segments. Training data segments are numbered
sequentially, while testing data are in random order. Within folders data 
segments are stored in .mat files as follows:

    I_J_K.mat - the Jth training data segment corresponding to the Kth class
    (K=0 for interictal, K=1 for preictal) for the Ith patient (there are 3 patients).
    I_J.mat - the Jth testing data segment for the Ith patient.

Each .mat file contains a data structure, dataStruct, with fields as follows:

    data: a matrix of iEEG sample values arranged row x column as time sample x electrode.
    nSamplesSegment: total number of time samples (number of rows in the data field).
    iEEGsamplingRate: data sampling rate, i.e. the number of data samples representing 1 second of EEG data. 
    channelIndices: an array of the electrode indexes corresponding to the columns in the data field.
    sequence: the index of the data segment within the one hour series of clips (see below). 
        For example, 1_12_1.mat has a sequence number of 6, and represents the iEEG data from 
        50 to 60 minutes into the preictal data. This field only appears in training data.

Data courtesy of epilepsyecosystem.org

"""

"""
Citations

A. Temko, E. Thomas, W. Marnane, G. Lightbody, G. Boylan,
EEG-based neonatal seizure detection with Support Vector Machines,
Clinical Neurophysiology,
Volume 122, Issue 3,
2011,
Pages 464-473,
ISSN 1388-2457,

"""

"""
General idea

Generate features from each 10 minute EEG segment and with these features try 
to predict whether it contains a seizure. i.e. the temporal data will not be 
fed into the ML algorithm, only the features generated from it.

Downsampling :
    The frequency of seizure EEGs ranges from 0.5HZ to 13HZ so we can downsample to 26Hz
"""

from os import listdir
from os.path import isfile, join

import warnings
warnings.filterwarnings('ignore')

import scipy.signal
import scipy.stats
import scipy.io as sio
from scipy.integrate import simps

import numpy as np

import sklearn.preprocessing as preprocessing

#%%  

train_data_path = 'C:/Users/gijsb/OneDrive/Documents/epilepsy_neurovista_data/Trainbis'
PATIENTS = [2] # [1, 2, 3]
SAMPLING_FREQUENCY = 400
DOWNSAMPLING_RATIO = 5
CHANNELS = range(0,16)
FREQ_RES = 0.25 #ToDO remove, not fixed value
BANDS = [0.1,1,4,8,12,30,70]
HIGHRES_BANDS = [0.1,1,4,8,12,30,70, 180]

#%%

filelist = [join(train_data_path, f) for f in listdir(train_data_path) if isfile(join(train_data_path, f))]

def load_mat(mat_file_path):
    try :
        return (sio.loadmat(mat_file_path)['data']).T
        
    except Exception :
        print('error reading .mat file')
        print(mat_file_path)
        return np.zeros((16, 240000))

#%%

def zero_crossings(data):
    pos = data > 0
    return (pos[:-1] & ~pos[1:]).nonzero()[0].shape[0]

def band_energy(f, psd, low_f, high_f):
        idx_delta = np.logical_and(f >= low_f, f <= high_f) # Find intersecting values in frequency vector
        freq_res = f[1] - f[0] # The frequency resolution is the size of each frequency bin
        return simps(psd[idx_delta], dx = freq_res) # Compute the absolute power by approximating the integral
    
def total_energy(segment_downsampled):
    window = SAMPLING_FREQUENCY / DOWNSAMPLING_RATIO * 4 #From litterature the lowest frequencies of interest in a EEG is 0.5Hz so we need to keep our resolution at 0.25Hz hence a 4 second window cf.Nyquist
    f, psd = scipy.signal.welch(segment_downsampled, fs = SAMPLING_FREQUENCY/DOWNSAMPLING_RATIO , nperseg = window)
    return psd.sum()

def highres_total_energy(segment):
    window = SAMPLING_FREQUENCY * 4
    f, psd = scipy.signal.welch(segment, fs = SAMPLING_FREQUENCY, nperseg = window)
    return psd.sum()

#ToDo Harmonise techniques

def _embed(x, order=3, delay=1):# credits to raphaelvallat
    """Time-delay embedding.
    Parameters
    ----------
    x : 1d-array
        Time series, of shape (n_times)
    order : int
        Embedding dimension (order).
    delay : int
        Delay.
    Returns
    -------
    embedded : ndarray
        Embedded time-series, of shape (n_times - (order - 1) * delay, order)
    """
    N = len(x)
    if order * delay > N:
        raise ValueError("Error: order * delay should be lower than x.size")
    if delay < 1:
        raise ValueError("Delay has to be at least 1.")
    if order < 2:
        raise ValueError("Order has to be at least 2.")
    Y = np.zeros((order, N - (order - 1) * delay))
    for i in range(order):
        Y[i] = x[i * delay:i * delay + Y.shape[1]]
    return Y.T

def svd_entropy(x, order=3, delay=1, normalize=False):
    x = np.array(x)
    mat = _embed(x, order=order, delay=delay)
    W = np.linalg.svd(mat, compute_uv=False)
    # Normalize the singular values
    W /= sum(W)
    svd_e = -np.multiply(W, np.log2(W)).sum()
    if normalize:
        svd_e /= np.log2(order)
    return svd_e

#%%


# Loop over files in filelist to generate features
#ToDo make a clean iteratin over each patients file
for patient in [2]:
    
    counter = 0
    for mat_file_path in filelist[0:1]:
        print(counter)
        counter += 1
        # Array that will contain index and features values, we will stack these to make X_Train
        index = []
        features = []
        
        #Load file & normalise
        data = load_mat(mat_file_path)
        data = preprocessing.scale(data, axis=1,with_std=True)
        data_downsampled = scipy.signal.decimate(data, 5, zero_phase = True)
        
        #accross channels features on full data
        correlation_matrix = np.corrcoef(data)
        correlation_matrix = np.nan_to_num(correlation_matrix)
        triup_index = np.triu_indices(16, k=1) #take only values in upper triangle to avoid redundancy    
        for i,j in zip(triup_index[0], triup_index[1]):
            features.append(correlation_matrix[i][j])
            index.append(f'correlation_{i}-{j}')
            
        eigenvals = np.linalg.eigvals(correlation_matrix)
        eigenvals = np.nan_to_num(eigenvals)
        eigenvals = np.real(eigenvals)
        for i in CHANNELS:
            features.append(eigenvals[i])
            index.append(f'eigenval_{i}')
        
        summed_energy = total_energy(data_downsampled) #summed across all channels and frequencies
        features.append(summed_energy)
        index.append(f'summed_energy')   
        
        #Per channel features
        #ToDo work on all channels in parrallel as one matrix, vectorise all of it
        for c in CHANNELS:
            # Create necessary functions
            data_channel = data_downsampled[c]
            diff1 = np.diff(data_channel, n=1)
            diff2 = np.diff(data_channel, n=2)
            
            # Simple features
            std = np.std(data_channel)
            features.append(std)
            index.append(f'std_{c}')
            
            skew = scipy.stats.skew(data_channel)
            features.append(skew)
            index.append(f'skew_{c}')
            
            kurt = scipy.stats.kurtosis(data_channel)
            features.append(kurt)
            index.append(f'kurt_{c}')
            
            zeros = zero_crossings(data_channel)
            features.append(zeros)
            index.append(f'zeros_{c}')
            
            #RMS = np.sqrt(data_channel**2.mean())
            
            # Differential features
            mobility = np.std(diff1)/np.std(data_channel)
            features.append(mobility)
            index.append(f'mobility_{c}')
            
            complexity = (np.std(diff2) * np.std(diff2))/np.std(diff1)
            features.append(complexity)
            index.append(f'complexity_{c}')
            
            zeros_diff1 = zero_crossings(diff1)
            features.append(zeros_diff1)
            index.append(f'zeros_diff1_{c}')
            
            zeros_diff2 = zero_crossings(diff2)
            features.append(zeros_diff2)
            index.append(f'zeros_diff2_{c}')
            
            std_diff1 = np.std(diff1)
            features.append(std_diff1)
            index.append(f'std_diff1_{c}')
            
            std_diff2 = np.std(diff2)
            features.append(std_diff2)
            index.append(f'std_diff2_{c}')
            
            # Frequency features
            
            ##Use welch method to approcimate energies per frequency subdivision
            window = (SAMPLING_FREQUENCY / DOWNSAMPLING_RATIO) * 4 #From litterature the lowest frequencies of interest in a EEG is 0.5Hz so we need to keep our resolution at 0.25Hz hence a 4 second window cf.Nyquist
            f, psd = scipy.signal.welch(data_channel, fs = 80, nperseg = window)
            psd = np.nan_to_num(psd)
            
            ## Total summed energy
            channel_energy = band_energy(f, psd, 0.1, 40)
            features.append(channel_energy)
            index.append(f'channel_{c}_energy')
            
            ## Normalised summed energy
            normalised_energy = channel_energy / summed_energy
            features.append(normalised_energy)
            index.append(f'normalised_energy_{c}')
            
            ## Peak frequency
            peak_frequency = f[np.argmax(psd)]
            features.append(peak_frequency)
            index.append(f'peak_frequency_{c}') 
            
            ## Normalised_summed energy per band
            for k in range(len(BANDS)-1):
                energy = band_energy(f, psd, BANDS[k], BANDS[k+1])
                normalised_band_energy = energy / channel_energy
                features.append(normalised_band_energy)
                index.append(f'normalised_band_energy_{c}_{k}')
                
            ## Spectralentropy
            psd_norm = np.divide(psd, psd.sum())
            spectral_entropy = -np.multiply(psd_norm, np.log2(psd_norm)).sum()
            #spectral_entropy /= np.log2(psd_norm.size) #uncomment to normalise entropy
            features.append(spectral_entropy)
            index.append(f'spectral_entropy_{c}')
            
            ## SVD entropy
            entropy = svd_entropy(data_channel, order=3, delay=1, normalize=False)
            features.append(entropy)
            index.append(f'svd_entropy_{c}')
            
            # Highres features
            highres_channel_energy = highres_total_energy(data)
            f, psd = scipy.signal.welch(data, fs = 400, nperseg = SAMPLING_FREQUENCY*4)
            psd = np.nan_to_num(psd)
            full_psd_sum = psd.sum()/10 #for normalisation purposed
            
            for j in range(19):
                data_segment = data_channel[j*30*SAMPLING_FREQUENCY : (j+1)*30*SAMPLING_FREQUENCY]
                f_segment, psd_segment = scipy.signal.welch(data_segment, fs=SAMPLING_FREQUENCY, nperseg = SAMPLING_FREQUENCY*4)
                psd_segment = np.nan_to_num(psd_segment)
                
                for k in range(len(HIGHRES_BANDS)-1):
                    normalised_band_energy = psd_segment[(f_segment > HIGHRES_BANDS[k]) & (f_segment < HIGHRES_BANDS[k+1])].sum()
                    features.append(normalised_band_energy)
                    index.append(f'normalised_band_energy_{c}_{k}_{j}')
            
            #print(np.asarray(features).shape)
            
        #if counter == 1:
            #X_train = np.zeros((1, len(features)))
            #X_train = np.array(features)
            # ToDo reset this once patient 2 is done
        #    X_train = np.load('C:/Users/gijsb/Workspace/Epilepsy prediction EEG/x_train_pat3_1359.npy')
        #    X_train = np.vstack((X_train, np.array(features)))
            
        #if counter > 1:
        #    X_train = np.vstack((X_train, np.array(features)))
            
        #ToDo add logging
        #print(X_train.shape)

#%%
     
#X_train_temp = np.vstack((X_train[0:1359][:], np.array(features)))
#X_train_temp = np.vstack((X_train_temp, X_train[1359:][:]))
#X_train = X_train_temp
