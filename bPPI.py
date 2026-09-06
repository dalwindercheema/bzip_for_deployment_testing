import os
import sys
import numpy
import torch
import configparser
import pandas as pd
import warnings
import math

warnings.simplefilter("always")

from bin.vocab import new_amino_acid_vocab
from bin.encode_data import transform_bzip_seqs, aminoseqdataset
from bin.bzip_models import BZIP_MOTIF, BZIP_INTERACTION



def sigmoid(x):
    return 1 / (1 + math.exp(-x))

    
def evaluate_motifs(model, seq, addinfo, seg, device):
    model.eval()
    with torch.no_grad():
        logits_clsf, _ = model(seq, 
                               addinfo,                               
                               seg,
                              )
        #print(logits_clsf)
        logits_clsf = logits_clsf.detach().cpu().numpy()
        return logits_clsf

def read_sequence(seq_path):
    seq = ""
    with open(seq_path, "r", encoding="UTF-8") as f:
        for line in f:
            if not line.startswith(">"):
                seq += line.strip()
    return seq


def find_motifs(protA, protB, bzip_motif_detector, bzip_interaction_detector, mvocab, ovocab, device, MAXL = 92, max_batch_size = 256):
    max_to_search_A = len(protA) - 29
    max_to_search_B = len(protB) - 29
    indices_A = [i for i, ch in enumerate(protA[:max_to_search_A]) if ch == 'N']
    indices_B = [i for i, ch in enumerate(protB[:max_to_search_B]) if ch == 'N']
    n_pairs = []
    for A in indices_A:    
        for B in indices_B:
            motif_A = protA[A: (A+92)]
            motif_B = protB[B: (B+92)]
            n_pairs.append((motif_A, motif_B))

    bdata_tf = transform_bzip_seqs(n_pairs, mvocab, ovocab, am_acid, MAXL)
    am_info, add_info, seg_info  = bdata_tf.transform(False)
    am_info = torch.Tensor(am_info).to(dtype=torch.long, device=device)
    add_info = torch.Tensor(add_info).to(dtype=torch.long, device=device)
    seg_info = torch.Tensor(seg_info).to(dtype=torch.long, device=device)
    print(am_info.shape)
    if(am_info.shape[0] > max_batch_size ):
        logits_clsf = []
        for i in range(0, am_info.shape[0], max_batch_size):
            tmp_logits_clsf = evaluate_motifs(bzip_motif_detector, am_info[i:i+max_batch_size, :], add_info[i:i+max_batch_size, :, :], seg_info[i:i+max_batch_size, :], device)
            logits_clsf.append(tmp_logits_clsf)
        logits_clsf = numpy.concat(logits_clsf)
    else:
        logits_clsf = evaluate_motifs(bzip_motif_detector, am_info, add_info, seg_info, device)
    sigmoid_prob = numpy.array([sigmoid(logits_clsf[i, 1]) for i in range(logits_clsf.shape[0])])
    sorted_indices = numpy.argsort(sigmoid_prob)[::-1]
    high_prob_indices = sorted_indices[sigmoid_prob[sorted_indices] > 0.5]
    passed_pairs = high_prob_indices[:1]
    if( passed_pairs.shape[0] > 0 ):
        logits_clsf = evaluate_motifs(bzip_interaction_detector, seq[passed_pairs, :], addinfo[passed_pairs, :, :], seg[passed_pairs, :], device)
        sigmoid_prob = numpy.array([sigmoid(logits_clsf[i, 1]) for i in range(logits_clsf.shape[0])])
        sorted_indices = numpy.argsort(sigmoid_prob)[::-1]
        max_pred = sigmoid_prob[sorted_indices[0]]
        n_interaction_preds = len(sorted_indices[sigmoid_prob[sorted_indices] > 0.5])
        return len(high_prob_indices), n_interaction_preds, max_pred
    else:
        return 0, 0, 0


def bPPI_predict(seqA, seqB):
    bzip_motif_detector_loc = "/content/drive/My Drive/bzip/motifDectector.pt"
    bzip_interaction_detector_loc = "/content/drive/My Drive/bzip/GeneralInteraction.pt"
    
    from experiments import test_model
    config = configparser.ConfigParser()
    config.read('config')
    device = 'cuda'

    d_model = int(config['pretrain']['d_model'])
    n_heads = int(config['pretrain']['n_head'])
    n_layers = int(config['pretrain']['n_layer'])
    bzipmaxlen = 92
    n_segments = 2 + (bzipmaxlen-15)//7
    
    mvocab, ovocab, am_acid = new_amino_acid_vocab()
    vocab_size = len(mvocab)
    ovocab_size = [len(ovocab.get(k)) for k in ovocab.keys()]
    
    bzip_motif_detector = BZIP_MOTIF(vocab_size, ovocab_size, n_segments, d_model, n_layers, n_heads)
    snap = torch.load(bzip_motif_detector_loc, weights_only=True)
    print('Loading motif detector model: Start')
    bzip_motif_detector.load_state_dict(snap)
    print('Loading motif detector model: Done!')
    bzip_motif_detector = bzip_motif_detector.to(device)
    
    bzip_interaction_detector = BZIP_INTERACTION(vocab_size, ovocab_size, n_segments, d_model, n_layers, n_heads)
    snap = torch.load(bzip_interaction_detector_loc, weights_only=True)
    print('Loading motif interaction model: Start')
    bzip_interaction_detector.load_state_dict(snap)
    print('Loading motif interaction model: Done!')
    bzip_interaction_detector = bzip_interaction_detector.to(device)
    
    seqA = read_sequence(seqA)
    seqB = read_sequence(seqA)
    #seqA = read_sequence(f"/home/dsinghc/scratch/other_proj/bzip/raw_data/bPPI_data/{protA}.fa")
    #seqB = read_sequence(f"/home/dsinghc/scratch/other_proj/bzip/raw_data/bPPI_data/{protB}.fa")
    nmotifs, npreds, max_pred = find_motifs(seqA, seqB, bzip_motif_detector, bzip_interaction_detector, mvocab, ovocab, device)
    print(idx, protA, protB, nmotifs, npreds, max_pred)
    pred_outputs.append([protA, protB, curr_eval[2], max_pred])
    