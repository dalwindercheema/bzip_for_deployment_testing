#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 00:45:17 2025

@author: dsing243
"""
import numpy

def apply_mask(n_data, vocab, max_mask_counts):
    # 80% probability to choose mask and 20% original
    # From 80% probability of mask 25% will be replaced
    min_id_to_rep = vocab['[MASK]'] + 1 # Assuming mask is the last special symbol and 20 single amino-acids
    max_id_to_rep = len(vocab)
    MASK = '<mask>'
    CLS = '<cls>'
    amino_symbls = list(vocab.keys())
    masked_seqs = []
    masked_tokens = []
    masked_pos = []
    for xidx in range(len(n_data)):
        tmp_paired_seq = [CLS] + [(seqa,seqb) for seqa,seqb in  zip(n_data[xidx][0], n_data[xidx][1]) ]
        masked_tokens_seq = []
        if(max_mask_counts > 0):
            shuffled_ids = numpy.arange(1, len(tmp_paired_seq))
            numpy.random.shuffle(shuffled_ids)
            shuffled_ids = shuffled_ids[:max_mask_counts]
            shuffled_ids.sort()
            for pos in shuffled_ids:
                masked_tokens_seq.append(tmp_paired_seq[pos])
                if(numpy.random.rand() < 0.8):
                    if(numpy.random.rand() < 0.20):
                        idx_to_replace = numpy.random.randint(min_id_to_rep, max_id_to_rep)
                        rep_amino = amino_symbls[idx_to_replace]  # replace
                        tmp_paired_seq[pos] = rep_amino
                    else:
                        tmp_paired_seq[pos] = MASK
        
            masked_tokens.append(masked_tokens_seq)
            masked_pos.append(shuffled_ids)
        else:
            masked_tokens.append([])
            masked_pos.append([])
        masked_seqs.append(tmp_paired_seq)
    masked_pos = numpy.array(masked_pos)
    return masked_seqs, masked_tokens, masked_pos