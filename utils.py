# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 15:17:16 2021

@author: gijsb
"""

import numpy as np
from scipy.integrate import simps
import scipy.io as sio
import scipy.stats
import scipy.signal


import warnings
warnings.filterwarnings('ignore')
import logging


#%% Parameters


SAMPLING_FREQUENCY = 400
DOWNSAMPLING_RATIO = 5


#%% Utility functions


def load_mat(mat_file_path):
    try:
        data = (sio.loadmat(mat_file_path)['data']).T
        logging.debug('data loaded')
        return data

    except Exception:
        warnings.warn(f'error reading .mat file {mat_file_path}')
        return np.zeros((16, 240000))
    
    
def add_feature(feat_value, feat_name, index, features):
    features = features.append(feat_value)
    index.append(feat_name)
    return (index, features)
    

#%% Functions used to generate features


def zero_crossings(data):
    pos = data > 0
    return (pos[:-1] & ~pos[1:]).nonzero()[0].shape[0]


def band_energy(f, psd, low_f, high_f):
    # Find intersecting values in frequency vector
    idx_delta = np.logical_and(f >= low_f, f <= high_f)
    # The frequency resolution is the size of each frequency bin
    freq_res = f[1] - f[0]
    # Compute the absolute power by approximating the integral
    return simps(psd[idx_delta], dx=freq_res)


def total_energy(segment_downsampled):
    # From litterature the lowest frequencies of interest in a EEG is 0.5Hz so we need to keep our resolution at 0.25Hz hence a 4 second window cf.Nyquist
    window = SAMPLING_FREQUENCY / DOWNSAMPLING_RATIO * 4
    f, psd = scipy.signal.welch(
        segment_downsampled, fs=SAMPLING_FREQUENCY/DOWNSAMPLING_RATIO, nperseg=window)
    return psd.sum()


def highres_total_energy(segment):
    window = SAMPLING_FREQUENCY * 4
    f, psd = scipy.signal.welch(segment, fs=SAMPLING_FREQUENCY, nperseg=window)
    return psd.sum()

#TODO Harmonise techniques

def _embed(x, order=3, delay=1):  # credits to raphaelvallat
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


# TODO add shannon entropy