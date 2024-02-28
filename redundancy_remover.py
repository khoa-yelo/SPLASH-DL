import numpy as np

def group_shifted_seqs(seq_dict):
    seq_clust_alls = []
    for seq, index in seq_dict.items():
        seq_clust = []
        pad_length = int(0.1 * len(seq))
        truncated_seq = seq[pad_length: len(seq) - pad_length]
        for seq_compare, index in seq_dict.items():
            found = seq_compare.find(truncated_seq)
            if found > -1:
                seq_clust.append(index)
        seq_clust_alls.append(seq_clust)
    return seq_clust_alls

def choose_longest_seq(seq_clusts, seq_dict):
    reversed_dict = {value: key for key, value in seq_dict.items()}
    minimal_seqs = []
    indices = []
    for clust in seq_clusts:
        max_i = np.argmax([len(reversed_dict[i]) for i in clust])
        minimal_seqs.append(reversed_dict[clust[max_i]])
        indices.append(clust[max_i])
    return minimal_seqs, indices

def remove_duplicates(list_of_lists):
    set_of_tuples = set()
    unique_lists = []
    for sublist in list_of_lists:
        tuple_sublist = tuple(sublist)
        if tuple_sublist not in set_of_tuples:
            set_of_tuples.add(tuple_sublist)
            unique_lists.append(sublist)
    return unique_lists

def compress_seqs(seqs):
    curr_nums = [len(seqs)]
    minimal_seqs = seqs
    minimal_seqs = {i:s for s, i in enumerate(minimal_seqs)}
    while (len(curr_nums) < 2) or curr_nums[-1] != curr_nums[-2]:
        grouped_seq = group_shifted_seqs(minimal_seqs)
        seq_clusts = remove_duplicates(grouped_seq)
        seqs_only, indices = choose_longest_seq(seq_clusts, minimal_seqs)
        minimal_seqs = {i:s for s, i in tuple(zip(indices, seqs_only))}
        curr_nums.append(len(minimal_seqs))
        print("Reduce from {} to {}".format(curr_nums[-2], curr_nums[-1]))
    return seqs_only, indices