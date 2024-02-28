"""
Module contains functions to extract DNA features
"""

import numpy as np
from stats_utils import calculate_entropy
from seq_utils import reverse_complement, complement


def gc_content(seq):
    """
    Function to calculate the GC content of a DNA sequence
    """
    gc = seq.count("G") + seq.count("C")
    return gc / len(seq)


def calculate_transition_matrix(seq):
    """
    Function to calculate the transition matrix of a DNA sequence
    """
    transition_matrix = np.zeros((4, 4), dtype=float)
    nu2ind = {"A": 0, "C": 1, "G": 2, "T": 3}
    for i in range(len(seq) - 1):
        curr = seq[i]
        next = seq[i + 1]
        transition_matrix[nu2ind[curr]][nu2ind[next]] += 1
    # Normalize the matrix to represent probabilities
    padding = 0.0001
    row_sums = transition_matrix.sum(axis=1, keepdims=True) + padding
    transition_matrix /= row_sums
    return transition_matrix


def max_homopolymer_length(seq):
    assert len(seq) > 0, "Input sequence must not be empty"
    max_length = 1
    current_length = 1
    current_nucleotide = seq[0]
    for nucleotide in seq[1:]:
        if nucleotide == current_nucleotide:
            current_length += 1
        else:
            current_length = 1
            current_nucleotide = nucleotide
        max_length = max(max_length, current_length)
    return max_length


def calculate_seq_entropy(seq):
    """
    Function to calculate the entropy of a DNA sequence
    """
    p = np.zeros(4)
    for base in seq:
        if base == "A":
            p[0] += 1
        elif base == "C":
            p[1] += 1
        elif base == "G":
            p[2] += 1
        elif base == "T":
            p[3] += 1
    p /= len(seq)
    return calculate_entropy(p)



def generate_dotplot(seq1, seq2, window_size=1):
    """
    Function to generate a dotplot of two DNA sequences
    """
    dotplot = np.zeros((len(seq1), len(seq2)))
    for i in range(len(seq1)):
        for j in range(len(seq2)):
            if seq1[i : i + window_size] == seq2[j : j + window_size]:
                dotplot[i, j] = 1
    return dotplot


# Mutual information
def mutual_information(seq1, seq2):
    """
    Function to calculate the mutual information between two DNA sequences
    """
    p = np.zeros((4, 4))
    for i in range(len(seq1)):
        p[nu2ind[seq1[i]], nu2ind[seq2[i]]] += 1
    p /= len(seq1)
    p1 = p.sum(axis=1)
    p2 = p.sum(axis=0)
    mi = 0
    for i in range(4):
        for j in range(4):
            if p[i, j] > 0:
                mi += p[i, j] * np.log(p[i, j] / (p1[i] * p2[j]))
    return mi


# longest repeats (allowed mismatch)
# dot plot
# diff between 2 dot plot
# cross dotplot


# write tests for the function
def test_gc_content():
    """
    Function to test the gc_content function
    """
    assert gc_content("ATGC") == 0.5
    assert gc_content("AGCG") == 0.75
    assert gc_content("AGCT") == 0.5
    assert gc_content("AGCTATAG") == 0.375
    print("All tests passed")


test_gc_content()
