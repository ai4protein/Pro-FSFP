import torch
import pandas as pd
from fsfp import config
from .base import MutantSequenceData

foldseek_struc_vocab = "pynwrqhgdlvtmfsaeikc#"

class SaProtMutantData(MutantSequenceData):
    def __init__(self, protein, tokenizer, mask=False, device=None):
        super().__init__(protein, tokenizer, mask=mask, device=device)
        struc_seqs = pd.read_csv(config.struc_seq_path, index_col='protein')
        protein_name = '_'.join(protein['name'].split('_')[:2])
        combined_seq = struc_seqs.loc[protein_name, 'struc_sequence']
        combined_seq = combined_seq[protein['offset'] * 2: (protein['offset'] + 1022) * 2]
        self.sequences = [combined_seq]
    
    def collate(self, raw_batch):
        wt_aas, mt_aas, positions, scores, labels = zip(*raw_batch)
        inv_idx = torch.zeros(len(positions), dtype=torch.long, device=self.device)
        
        sequences = self.tokenizer(self.sequences, return_tensors='pt').to(self.device)
        positions = [[pos + 1 for pos in position] for position in positions]
        vocab = self.tokenizer.get_vocab()
        wt_aas = [[vocab[wt + foldseek_struc_vocab[0]] for wt in wts] for wts in wt_aas]
        mt_aas = [[vocab[mt + foldseek_struc_vocab[0]] for mt in mts] for mts in mt_aas]
        scores = torch.tensor(scores, device=self.device)
        labels = torch.tensor(labels, device=self.device)
        return dict(sequences=sequences,
                    positions=positions,
                    inv_seq_idx=inv_idx,
                    targets=scores,
                    labels=labels,
                    wt_aas=wt_aas,
                    mt_aas=mt_aas)

def saprot_zero_shot(model, batch):
    logits = model(**batch['sequences']).logits.squeeze(0)
    log_prob = torch.log_softmax(logits, dim=-1) # length * num_aa
    
    predicts = []
    for positions, wt_aas, mt_aas in zip(batch['positions'], batch['wt_aas'], batch['mt_aas']):
        predict = 0
        for pos, wt, mt in zip(positions, wt_aas, mt_aas):
            wt_prob = log_prob[pos, wt: wt + len(foldseek_struc_vocab)].mean()
            mt_prob = log_prob[pos, mt: mt + len(foldseek_struc_vocab)].mean()
            predict += (mt_prob - wt_prob)
        predicts.append(predict.unsqueeze(0))
    return torch.cat(predicts)
