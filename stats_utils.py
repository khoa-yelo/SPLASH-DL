"""
This module contains functions to calculate statistical metrics
Author: Khoa Hoang
Date: Feb 1st 2024
"""

import numpy as np


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
