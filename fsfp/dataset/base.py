import random
import math
import torch
from torch.utils.data import Dataset, DataLoader
from itertools import combinations

class ProteinSequenceData(Dataset):
    def __init__(self, sequences, tokenizer, device=None):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.device = device
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx]
    
    def collate(self, raw_batch):
        sequences = self.tokenizer(raw_batch, return_tensors='pt', padding=True, return_length=True)
        return sequences.to(self.device)

class MutantSequenceData(Dataset):
    def __init__(self, protein, tokenizer, mask=False, device=None):
        if mask:
            self.sequences = {}
            for positions in set(protein['df']['positions']):
                mutant = list(protein['wild_type'])
                for position in positions: # get masked mutant sequence
                    mutant[position] = '<mask>'
                self.sequences[positions] = ''.join(mutant)
        else:
            self.sequences = [protein['wild_type']]
        
        for key, value in protein['df'].items():
            setattr(self, key, value.to_list())
        self.tokenizer = tokenizer
        self.device = device
    
    def __len__(self):
        return len(self.positions)
    
    def __getitem__(self, idx):
        return self.wt_aas[idx], self.mt_aas[idx], self.positions[idx], self.DMS_score[idx], self.DMS_score_bin[idx]
    
    def collate(self, raw_batch):
        wt_aas, mt_aas, positions, scores, labels = zip(*raw_batch)
        
        if type(self.sequences) is dict: # identify duplicate positions, possibly multi-site
            unique_pos = {pos: i for i, pos in enumerate(set(positions))}
            inv_idx = torch.tensor([unique_pos[pos] for pos in positions], device=self.device)
            sequences = [self.sequences[pos] for pos in unique_pos.keys()]
        else:
            inv_idx = torch.zeros(len(positions), dtype=torch.long, device=self.device)
            sequences = self.sequences
        sequences = self.tokenizer(sequences, return_tensors='pt').to(self.device)
        
        positions = [torch.tensor(pos, device=self.device) + 1 for pos in positions]
        wt_aas = self.tokenizer(wt_aas, add_special_tokens=False)['input_ids']
        mt_aas = self.tokenizer(mt_aas, add_special_tokens=False)['input_ids']
        scores = torch.tensor(scores, device=self.device)
        labels = torch.tensor(labels, device=self.device)
        return dict(sequences=sequences,
                    inv_seq_idx=inv_idx,
                    wt_aas=wt_aas,
                    mt_aas=mt_aas,
                    positions=positions,
                    targets=scores,
                    labels=labels)

class RankingSequenceData(Dataset):
    def __init__(self, protein, tokenizer, mask=True, list_size=2, max_size=10000,
                 constructor=MutantSequenceData, device=None):
        self.mutant_data = constructor(protein, tokenizer, mask, device)
        self.list_size = list_size
        self.max_size = max_size
        self.device = device
        
        total = math.comb(len(self.mutant_data), list_size)
        if max_size > total: # iteration over all combinations
            self.comb_idx = list(combinations(range(len(self.mutant_data)), list_size))
        else: # numerous combinations, random select instead
            self.comb_idx = None
    
    def __len__(self):
        if self.comb_idx is not None:
            return len(self.comb_idx)
        else:
            return self.max_size
    
    def __getitem__(self, idx): # yield combination indices instead of real data
        if self.comb_idx is not None:
            return self.comb_idx[idx]
        else:
            return random.sample(range(len(self.mutant_data)), self.list_size)
    
    def collate(self, comb_idx): # identify duplicate elements among a batch of combinations
        comb_idx = torch.tensor(comb_idx, device=self.device)
        unique_mt, inv_idx = torch.unique(comb_idx, return_inverse=True)
        raw_batch = [self.mutant_data[i] for i in unique_mt]
        batch = self.mutant_data.collate(raw_batch)
        batch['inv_list_idx'] = inv_idx
        return batch

class MetaRankingSequenceData(Dataset):
    def __init__(self, protein_splits, tokenizer, adapt_batch_size, eval_batch_size,
                 adapt_steps=5, mask='train', list_size=2, training=True,
                 constructor=MutantSequenceData, device=None):
        self.support_iters = []
        self.query_iters = []
        for support, query in protein_splits:
            # random selection
            support = RankingSequenceData(support, tokenizer,
                                          mask=mask in {'train', 'all'},
                                          list_size=list_size,
                                          max_size=adapt_steps * adapt_batch_size,
                                          constructor=constructor,
                                          device=device)
            support_iter = DataLoader(support,
                                      batch_size=adapt_batch_size,
                                      shuffle=True,
                                      collate_fn=support.collate)
            self.support_iters.append(support_iter)
            
            if training:
                query = RankingSequenceData(query, tokenizer,
                                            mask=mask in {'train', 'all'},
                                            list_size=list_size,
                                            max_size=eval_batch_size,
                                            constructor=constructor,
                                            device=device)
            else:
                query = constructor(query, tokenizer, mask=mask in {'eval', 'all'}, device=device)
            query_iter = DataLoader(query,
                                     batch_size=eval_batch_size,
                                     collate_fn=query.collate)
            self.query_iters.append(query_iter)
        
    def __len__(self):
        return len(self.query_iters)
    
    def __getitem__(self, idx):
        adapt_batch = [batch for batch in self.support_iters[idx]]
        eval_batch = next(iter(self.query_iters[idx]))
        return adapt_batch, eval_batch
    
    def collate(self, raw_batch):
        adapt_batches, eval_batches = zip(*raw_batch)
        return dict(adapt_batches=adapt_batches,
                    eval_batches=eval_batches)
