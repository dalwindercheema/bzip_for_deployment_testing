import numpy
import torch
from torch.utils.data import Dataset

class transform_bzip_seqs():
    def __init__(self,  dataset, mvocab, ovocab, am_acid, maxlen):
        self.zipper_begin = 15  #0-base corrdinate
        self.mvocab = mvocab
        self.ovocab = ovocab
        self.prot = dataset
        self.am_acid = am_acid
        self.MASK = mvocab['[MASK]']
        self.CLS = mvocab['[CLS]']
        self.vocabsize = len(mvocab)
        pmaxlen = (maxlen-self.zipper_begin) // 7
        self.maxlen = pmaxlen * 7 + self.zipper_begin + 1
        self.other_info = ['chemical', 'charge', 'H_donor' ,'polarity', 
                           'hydropathy', 'volume']
        self.n_other_info = len(self.other_info)
        self.seq_x = numpy.zeros((len(self.prot), self.maxlen))
        self.other_seq_x = numpy.zeros((len(self.prot), self.maxlen, 6))
        self.seg_x = numpy.zeros((len(self.prot), self.maxlen))
        
    def get_id(self, ids):
        id_for_vb = self.mvocab.get(ids)
        if(id_for_vb == None):
            id_for_vb = self.mvocab.get((ids[1], ids[0]))
        return id_for_vb
    
    def get_other_id(self, other_id, ids):
        if(len(ids) > 1 ):
            new_ids_x = self.am_acid.loc[ids[0], other_id]
            new_ids_y = self.am_acid.loc[ids[1], other_id]
            id_for_vb = self.ovocab[other_id].get((new_ids_x, new_ids_y))
            if(id_for_vb == None):
                id_for_vb = self.ovocab[other_id].get((new_ids_y, new_ids_x))
        else:
            new_ids = self.am_acid.loc[ids[0], other_id]
            id_for_vb = self.ovocab[other_id].get(new_ids)
        return id_for_vb
    
    def get_seqids(self, prot, prot_idx):
        n_idx = -1
        for p_idx, nprot in enumerate(prot):
            if(nprot == '<cls>'):
                id_for_vb = self.CLS
                id_for_other_vb = self.CLS * numpy.ones(self.n_other_info)
                
                self.seq_x[prot_idx, p_idx] = id_for_vb
                self.other_seq_x[prot_idx, p_idx, :] = id_for_other_vb
                self.seg_x[prot_idx, p_idx] = 1
                
                n_idx += 1
                continue
            if(nprot == '<mask>'):
                id_for_vb = self.MASK
                id_for_other_vb = self.MASK * numpy.ones(self.n_other_info)
            else:
                id_for_vb = self.get_id(nprot)
                id_for_other_vb = numpy.zeros(self.n_other_info)
                for oidx, oinfo in enumerate(self.other_info):
                    id_for_other_vb[oidx] = self.get_other_id(oinfo, nprot)

            if(n_idx < self.zipper_begin):
                self.seq_x[prot_idx, p_idx] = id_for_vb
                self.other_seq_x[prot_idx, p_idx, :] = id_for_other_vb
                self.seg_x[prot_idx, p_idx] = 1
                n_idx += 1
            elif(n_idx >= self.zipper_begin and (n_idx-self.zipper_begin)%7 == 0):
                sid = ((n_idx-self.zipper_begin)//7) + 2
                self.seq_x[prot_idx, p_idx] = id_for_vb
                self.other_seq_x[prot_idx, p_idx, :] = id_for_other_vb
                self.seg_x[prot_idx, p_idx] = sid
                n_idx += 1
            else:
                self.seq_x[prot_idx, p_idx] = id_for_vb
                self.other_seq_x[prot_idx, p_idx, :] = id_for_other_vb
                self.seg_x[prot_idx, p_idx] = sid
                n_idx += 1
   
   
    def transform(self):
        for idx in range(len(self.prot)):
            tmp_paired_seq = ['<cls>'] + [(seqa,seqb) for seqa,seqb in  zip(self.prot[idx][0], self.prot[idx][1]) ]
            self.get_seqids(tmp_paired_seq, idx)
        return (self.seq_x, self.other_seq_x, self.seg_x)
    
