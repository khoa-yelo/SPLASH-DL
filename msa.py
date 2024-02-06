"""
This module contain MSA class and associated functions
Author: Khoa Hoang
Date: Fed 5th, 2024
"""

import numpy as np

from seq_utils import onehot
from stats_utils import calculate_entropy_from_counts
from dna_feat_extractor import generate_dotplot


class MSA:
    def __init__(self, seqs, aligned_seqs, alphabet_dict, counts):
        self.seqs = seqs
        self.aligned_seqs = aligned_seqs
        self.alphabet_dict = alphabet_dict
        self.counts = counts
        self.entropy = calculate_entropy_from_counts(self.counts)

    @property
    def prob_matrix(self):
        onehot_vect = onehot(self.aligned_seqs, self.alphabet_dict)
        return (onehot_vect.sum(dim=0).T / onehot_vect.sum(dim=[0, 2])).T

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
    def dotplot_highest_count_seq(self):
        highest_count_seq = self.seqs[np.argmax(self.counts)]
        return generate_dotplot(highest_count_seq, highest_count_seq)

    def __str__(self):
        return str(self.aligned_seqs)


if __name__ == "__main__":

    aligned_seqs = ["ATG--C", "ATGAAC", "ATGGGC"]
    seqs = ["ATGC", "ATGAAC", "ATGGGC"]
    alphabet_dict = {"p": 0, "-": 1, "A": 2, "C": 3, "G": 4, "T": 5}
    msa = MSA(seqs, aligned_seqs, alphabet_dict, [1, 2, 3])
    print(msa.prob_matrix)
    print(msa.pairwise_lev_dist)
    print(msa.max_lev_dist)
    print(msa.average_lev_dist)
    print(msa.dotplot_highest_count_seq)
    print(msa.entropy)
