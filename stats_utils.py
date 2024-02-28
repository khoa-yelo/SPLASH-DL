"""
This module contains functions to calculate statistical metrics
Author: Khoa Hoang
Date: Feb 1st 2024
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap

def calculate_entropy(p):
    """
    Function to calculate the entropy of a probability distribution
    """
    return -np.sum(p * np.log(p))


def calculate_entropy_from_counts(counts):
    """
    Function to calculate the entropy of a probability distribution
    """
    p = counts / np.sum(counts)
    return calculate_entropy(p)


def kl_divergence(p, q):
    """
    Function to calculate the Kullback-Leibler divergence between two probability distributions
    """
    return np.sum(p * np.log(p / q))

def pca(X, n_component = 2, scale = True):
    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    pca_reducer = PCA(n_component)
    X = pca_reducer.fit_transform(X)
    return X

def umap_reduce(X, n_component = 2, scale = False):
    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    umap_reducer = umap.UMAP()
    X = umap_reducer.fit_transform(X)
    return X
