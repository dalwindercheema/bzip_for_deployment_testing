import os
import sys
import numpy
import torch
import configparser
import argparse
import json
import warnings
import math
from huggingface_hub import hf_hub_download

from bin.vocab import new_amino_acid_vocab
from bin.encode_data import transform_bzip_seqs
from bin.bzip_models import BZIP_MOTIF, BZIP_INTERACTION

warnings.filterwarnings("ignore", category=DeprecationWarning, module="jupyter_client")

CONFIG_FILE = "bppi_config.json"

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

    
def evaluate_motifs(model, seq, addinfo, seg, device):
    model.eval()
    with torch.no_grad():
        logits_clsf, _ = model(seq, 
                               addinfo,                               
                               seg,
                              )
        logits_clsf = logits_clsf.detach().cpu().numpy()
        return logits_clsf

def read_sequence(seq_path):
    seq = ""
    with open(seq_path, "r", encoding="UTF-8") as f:
        for line in f:
            if not line.startswith(">"):
                seq += line.strip()
    return seq


def find_motifs(protA, protB, bzip_motif_detector, bzip_interaction_detector, mvocab, ovocab, am_acid, device, MAXL = 92, max_batch_size = 256):
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
    am_info, add_info, seg_info  = bdata_tf.transform()
    am_info = torch.Tensor(am_info).to(dtype=torch.long, device=device)
    add_info = torch.Tensor(add_info).to(dtype=torch.long, device=device)
    seg_info = torch.Tensor(seg_info).to(dtype=torch.long, device=device)
    if(am_info.shape[0] > max_batch_size ):
        logits_clsf = []
        for i in range(0, am_info.shape[0], max_batch_size):
            tmp_logits_clsf = evaluate_motifs(bzip_motif_detector, 
                                              am_info[i:i+max_batch_size, :], 
                                              add_info[i:i+max_batch_size, :, :], 
                                              seg_info[i:i+max_batch_size, :], 
                                              device)
            logits_clsf.append(tmp_logits_clsf)
        logits_clsf = numpy.concat(logits_clsf)
    else:
        logits_clsf = evaluate_motifs(bzip_motif_detector, 
                                      am_info, 
                                      add_info, 
                                      seg_info, 
                                      device)
                                      
    sigmoid_prob = numpy.array([sigmoid(logits_clsf[i, 1]) for i in range(logits_clsf.shape[0])])
    sorted_indices = numpy.argsort(sigmoid_prob)[::-1]
    high_prob_indices = sorted_indices[sigmoid_prob[sorted_indices] > 0.5]
    if( len(high_prob_indices) > 0 ):
        best_motif = high_prob_indices[0:1]
        logits_clsf = evaluate_motifs(bzip_interaction_detector, 
                                      am_info[best_motif, :], 
                                      add_info[best_motif, :, :], 
                                      seg_info[best_motif, :], 
                                      device)
                                      
        sigmoid_prob = sigmoid(logits_clsf[0, 1])
        sigmoid_prob = numpy.round(sigmoid_prob, 2)
        return len(high_prob_indices), sigmoid_prob
    else:
        return 0, 0


def save_config(motif_path, interaction_path):
    with open(CONFIG_FILE, "w") as f:
        json.dump({
                   "motif_detector": motif_path,
                   "interaction_detector": interaction_path
                   }, f)

def load_config():
    if not os.path.exists(CONFIG_FILE):
        print("Error: Weights not found!", file=sys.stderr)
        print("Please run the download command first:", file=sys.stderr)
        print("  python predict.py --download General", file=sys.stderr)
        sys.exit(1)
    with open(CONFIG_FILE, "r") as f:
        return json.load(f)
        
_DOWNLOADED_MODELS = {
    "motif_detector": None,
    "interaction_detector": None
}

def download_models(PPI_TYPE="General"):
    model_mapping = {
                     "General": "GeneralInteraction.pt",
                     "Temperature4C": "Temperature_4.pt",
                     "Temperature21C": "Temperature_21.pt",
                     "Temperature37C": "Temperature_37.pt"
                    }
    
    if PPI_TYPE not in model_mapping:
        raise ValueError(f"Unknown PPI_TYPE selected. Choose from: {list(model_mapping.keys())}")
        
    filename = model_mapping[PPI_TYPE]
    
    print("Downloading motif detector... ")
    motif_detector_path = hf_hub_download(repo_id="dalwindercheema/bPPI", 
                                          filename="motifDectector.pt"
                                          )
                                                           
    print(f"Downloading {PPI_TYPE} interaction detector ({filename})... ")
    
    interaction_path = hf_hub_download(repo_id="dalwindercheema/bPPI", 
                                       filename=filename
                                       )
                                       
    print("All weights downloaded and loaded successfully!")
    save_config(motif_detector_path, interaction_path)



def bPPI_predict(seqA, seqB, debug = False):
    config = load_config()
    bzip_motif_detector_loc = config["motif_detector"]
    bzip_interaction_predictor_loc = config["interaction_detector"]
    
    print(f"Predicting with {bzip_interaction_predictor_loc.split('/')[-1].replace('.pt', '')}")
    config = configparser.ConfigParser()
    config.read('config')
    device = 'cpu'

    d_model = int(config['pretrain']['d_model'])
    n_heads = int(config['pretrain']['n_head'])
    n_layers = int(config['pretrain']['n_layer'])
    bzipmaxlen = 92
    n_segments = 2 + (bzipmaxlen-15)//7
    
    mvocab, ovocab, am_acid = new_amino_acid_vocab()
    vocab_size = len(mvocab)
    ovocab_size = [len(ovocab.get(k)) for k in ovocab.keys()]
    
    bzip_motif_detector = BZIP_MOTIF(vocab_size, ovocab_size, n_segments, d_model, n_layers, n_heads)
    snap = torch.load(bzip_motif_detector_loc, weights_only=True, map_location=torch.device('cpu'))
    bzip_motif_detector.load_state_dict(snap)
    bzip_motif_detector = bzip_motif_detector.to(device)
    
    bzip_interaction_detector = BZIP_INTERACTION(vocab_size, ovocab_size, n_segments, d_model, n_layers, n_heads)
    snap = torch.load(bzip_interaction_predictor_loc, weights_only=True, map_location=torch.device('cpu'))
    bzip_interaction_detector.load_state_dict(snap)
    bzip_interaction_detector = bzip_interaction_detector.to(device)
    
    seqA = read_sequence(seqA)
    seqB = read_sequence(seqB)
    nmotifs, max_pred = find_motifs(seqA, seqB, bzip_motif_detector, bzip_interaction_detector, mvocab, ovocab, am_acid, device)
    
    print(f"Motifs detected: {nmotifs}")
    print(f"Interaction Probability: {max_pred}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict bZIP protein protein interaction (bPPI)")
    
    parser.add_argument("--download", type=str, metavar="MODEL_TYPE",
                        choices=["General", "Temperature4C", "Temperature21C", "Temperature37C"],
                        help="Download weights. Options: General, Temperature4C, Temperature 21C, Temperature 37C")
    

    parser.add_argument("--seqA", type=str, metavar="PATH", help="FASTA file for sequence A")
    parser.add_argument("--seqB", type=str, metavar="PATH", help="FASTA file for sequence B")

    args = parser.parse_args()

    # Routing based on which flags the user provided
    if args.download:
        download_models(args.download)
    elif args.seqA and args.seqB:
        score = bPPI_predict(args.seqA, args.seqB)
    else:
        parser.print_help()
        print("\n Error: Please specify either --download OR both --seqA and --seqB flags.", file=sys.stderr)
        sys.exit(1)
      