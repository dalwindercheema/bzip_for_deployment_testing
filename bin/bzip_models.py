import torch
import torch.nn as nn
import math
import numpy as np
from einops import rearrange, repeat

class TEncoderLayer(nn.TransformerEncoderLayer):
    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        attn_output, attn_weights = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=True,           
            average_attn_weights=False   
        )
        x = self.norm1(src + self.dropout1(attn_output))
        x = self.norm2(x + self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(x))))))
        return x, attn_weights
    
    
def get_attn_pad_mask(seq_k):
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.eq(0)
    return pad_attn_mask
    

def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
                
class Embedding(nn.Module):
    def __init__(self, vocab_size, ovocab_size, n_segments, d_model, emb_dropout=0):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.chem_embed = nn.Embedding(ovocab_size[0], d_model, padding_idx=0)  # chemistry embedding
        self.chr_embed = nn.Embedding(ovocab_size[1], d_model, padding_idx=0)  # charge embedding
        self.hdon_embed = nn.Embedding(ovocab_size[2], d_model, padding_idx=0)  # H_donor embedding
        self.pol_embed = nn.Embedding(ovocab_size[3], d_model, padding_idx=0)  # polarity embedding
        self.phd_embed = nn.Embedding(ovocab_size[4], d_model, padding_idx=0)  # phydro embedding
        self.vol_embed = nn.Embedding(ovocab_size[5], d_model, padding_idx=0)  # volume embedding
        self.seg_embed = nn.Embedding(n_segments, d_model, padding_idx=0)  # bzip location embedding
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(emb_dropout)

    def forward(self, x, ox, seg):
        tok = self.tok_embed(x)
        seg = self.seg_embed(seg)
        chem = self.chem_embed(ox[:,:,0])
        chr = self.chr_embed(ox[:,:,1])
        hdon = self.hdon_embed(ox[:,:,2])
        pol = self.pol_embed(ox[:,:,3])
        phd = self.phd_embed(ox[:,:,4])
        vol = self.vol_embed(ox[:,:,5])
        embedding = tok + seg + chem + chr + hdon + pol + phd + vol
        embedding = self.dropout(self.norm(embedding))
        return embedding

class BZIP_MOTIF(nn.Module):
    def __init__(self, vocab_size, ovocab_size, n_segments, d_model, n_layers, n_heads):
        super(BZIP_MOTIF, self).__init__()
        classes = 2
        self.embedding = Embedding(vocab_size, ovocab_size, n_segments, d_model)
        self.encoder = TEncoderLayer(d_model=d_model, 
                                    nhead=n_heads, 
                                    dim_feedforward = d_model*4,
                                    activation = 'gelu',
                                    dropout = 0.2, 
                                    batch_first=True)
        
        self.encoder = nn.TransformerEncoder(self.encoder, 
                                             num_layers=n_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(0.01),
            nn.LayerNorm(d_model),
            nn.Dropout(0.2),
            nn.Linear(d_model, classes)
        )
        self.classifier.apply(weights_init)
        self.norm = nn.LayerNorm(d_model)
        self.amino = nn.Linear(d_model, vocab_size, bias=False)
        self.chem = nn.Linear(d_model, ovocab_size[0], bias=False)
        self.chr = nn.Linear(d_model, ovocab_size[1], bias=False)
        self.hdon = nn.Linear(d_model, ovocab_size[2], bias=False)
        self.pol = nn.Linear(d_model, ovocab_size[3], bias=False)
        self.phd = nn.Linear(d_model, ovocab_size[4], bias=False)
        self.vol = nn.Linear(d_model, ovocab_size[5], bias=False)
    
    def forward(self, seq, addinfo, seg):
        output = self.embedding(seq, addinfo, seg)
        enc_self_attn_mask = get_attn_pad_mask(seq)
        for tlayer in self.encoder.layers:
            output, enc_self_attn = tlayer(output, src_key_padding_mask = enc_self_attn_mask)        
        cls_output = output[:, 0, :]
        cls_output = self.norm(cls_output)
        logits_clsf = self.classifier(cls_output) 
        return logits_clsf, enc_self_attn

class BZIP_INTERACTION(nn.Module):
    def __init__(self, vocab_size, ovocab_size, n_segments, d_model, n_layers, n_heads):
        super(BZIP_INTERACTION, self).__init__()
        classes = 2
        self.embedding = Embedding(vocab_size, ovocab_size, n_segments, d_model)
        self.encoder = TEncoderLayer(d_model=d_model, 
                                    nhead=n_heads, 
                                    dim_feedforward = d_model*4,
                                    activation = 'gelu',
                                    dropout = 0.2, 
                                    batch_first=True)
        
        self.encoder = nn.TransformerEncoder(self.encoder, 
                                             num_layers=n_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(0.01),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, classes)
        )
        self.norm = nn.LayerNorm(d_model)
        self.classifier.apply(weights_init)

    
    def forward(self, seq, addinfo, seg):
        output = self.embedding(seq, addinfo, seg)
        enc_self_attn_mask = get_attn_pad_mask(seq)
        
        for tlayer in self.encoder.layers:
            output, enc_self_attn = tlayer(output, src_key_padding_mask = enc_self_attn_mask)
        cls_output = output[:, 0, :]
        cls_output = self.norm(cls_output)
        logits_clsf = self.classifier(cls_output)
        return logits_clsf, enc_self_attn