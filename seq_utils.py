"""
This module contains utils functions for DNA sequence analysis
Author: Khoa Hoang
Date: Feb 1st 2024
"""

import numpy as np
import torch


def validate_seq(dna_seq):
    """
    Validates the sequence to contain only ATGC characters
    """
    tmp_seq = dna_seq.upper()
    for nuc in tmp_seq:
        if nuc not in "ATGC":
            return False
    return tmp_seq


def complement(seq):
    """
    Function to calculate the complement of a DNA sequence
    """
    comp = {"A": "T", "T": "A", "G": "C", "C": "G"}
    return "".join([comp[base] for base in seq])


def reverse_complement(seq):
    """
    Function to calculate the reverse complement of a DNA sequence
    """
    return complement(seq[::-1])


lab2onehot = lambda seq, alphabet_dict: torch.eye(len(alphabet_dict))[seq]
seq2lab = lambda seq, alphabet_dict: torch.tensor([alphabet_dict[i] for i in seq])


def label(seqs, alphabet_dict):
    assert any(seqs), "empty sequence input"
    assert any([len(seq) != len(seq[0]) for seq in seqs])
    encoded_seqs = torch.ones((len(seqs), len(seqs[0])))
    for i, seq in enumerate(seqs):
        encoded_seq = torch.tensor([alphabet_dict[a] for a in seq])
        encoded_seqs[i, :] = encoded_seq
    return encoded_seqs


def onehot(seqs, alphabet_dict):
    assert any(seqs), "empty sequence input"
    assert any([len(seq) != len(seq[0]) for seq in seqs])
    encoded_seqs = torch.ones((len(seqs), len(seqs[0]), len(alphabet_dict)))
    for i, seq in enumerate(seqs):
        encoded_seq = seq2lab(seq, alphabet_dict)
        encoded_seq_onehot = lab2onehot(encoded_seq, alphabet_dict)
        encoded_seqs[i, :, :] = encoded_seq_onehot
    return encoded_seqs
