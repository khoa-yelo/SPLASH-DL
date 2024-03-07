"""
This module contain MSA class and associated functions
Author: Khoa Hoang
Date: Fed 5th, 2024
"""

import numpy as np

from seq_utils import onehot, label
from stats_utils import calculate_entropy_from_counts
from dna_feat_extractor import generate_dotplot


class MSA:
    def __init__(self, anchor, seqs, aligned_seqs, alphabet_dict, counts):
        self.anchor = anchor
        self.seqs = seqs
        self.aligned_seqs = aligned_seqs
        self.alphabet_dict = alphabet_dict
        self.counts = counts
        self.entropy = calculate_entropy_from_counts(self.counts)
        self.lengths = [len(seq) for seq in self.seqs]

    @property
    def seq_image_label(self):
        label_vect = label(self.aligned_seqs, self.alphabet_dict)
        return label_vect

    @property
    def seq_image(self):
        onehot_vect = onehot(self.aligned_seqs, self.alphabet_dict)
        return onehot_vect

    @property
    def prob_matrix(self):
        return (self.seq_image.sum(dim=0).T / self.seq_image.sum(dim=[0, 2])).T

    @property
    def pairwise_lev_dist(self):
        n = len(self.aligned_seqs)
        m = len(self.aligned_seqs[0])
        dist = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dist[i, j] = sum(
                    [
                        self.aligned_seqs[i][k] != self.aligned_seqs[j][k]
                        for k in range(m)
                    ]
                )
        return dist

    @property
    def max_lev_dist(self):
        return np.max(self.pairwise_lev_dist)

    @property
    def average_lev_dist(self):
        return np.mean(self.pairwise_lev_dist)

    @property
    def highest_count_seq(self):
        return self.seqs[np.argmax(self.counts)]

    @property
    def dotplot_highest_count_seq(self):
        return generate_dotplot(self.highest_count_seq, self.highest_count_seq)

    @property
    def longest_seq(self):
        return max(self.seqs, key=len)

    @property
    def mean_length(self):
        return np.mean(self.lengths)

    @property
    def max_length(self):
        return np.max(self.lengths)

    @property
    def min_length(self):
        return np.min(self.lengths)

    def __str__(self):
        return str(self.aligned_seqs)


if __name__ == "__main__":

    aligned_seqs = ["ATG--C", "ATGAAC", "ATGGGC"]
    seqs = ["ATGC", "ATGAAC", "ATGGGC"]
    alphabet_dict = {"-": 0, "A": 1, "C": 2, "G": 3, "T": 4}
    msa = MSA("AA", seqs, aligned_seqs, alphabet_dict, [1, 2, 3])
    print(msa.anchor)
    print(msa.prob_matrix)
    print(msa.pairwise_lev_dist)
    print(msa.max_lev_dist)
    print(msa.average_lev_dist)
    print(msa.dotplot_highest_count_seq)
    print(msa.entropy)
    print(msa.lengths)
    print(msa.mean_length)
    print(onehot(msa.aligned_seqs, msa.alphabet_dict).shape)
