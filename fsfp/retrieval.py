import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.functional import pairwise_distance
from transformers import EsmTokenizer, EsmForMaskedLM
from tqdm import tqdm
from . import config
from .dataset.base import ProteinSequenceData
from .utils.data import make_dir

class Protein2Vector():
    def __init__(self, model, pooling='average', hidden_fn=None):
        self.model = model
        self.pooling = pooling
        self.hidden_fn = hidden_fn
    
    def __call__(self, data_iter):
        results = []
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(data_iter, desc='Caching model outputs...'):
                lengths = batch.pop('length')
                if self.hidden_fn is None:
                    hiddens = self.model(**batch).hidden_states[-1]
                else:
                    hiddens = self.hidden_fn(self.model, batch)
                
                if self.pooling == 'average':
                    result = hiddens.sum(1).div(lengths.unsqueeze(1))
                elif self.pooling == 'max':
                    result = hiddens.max(1)
                else:
                    result = hiddens[:,-1,:]
                results.append(result.to('cpu'))
            results = torch.cat(results)
        return results

class Retriever():
    def __init__(self, parsed_args, data_constructor=ProteinSequenceData, hidden_fn=None):
        self.args = parsed_args
        self.device = 'cpu' if parsed_args.force_cpu or not torch.cuda.is_available() else 'cuda'
        self.data_constructor = data_constructor
        self.hidden_fn = hidden_fn
    
    def get_base_model(self):
        model_name = config.model_dir[self.args.model]
        model = EsmForMaskedLM.from_pretrained(model_name, output_hidden_states=True)
        tokenizer = EsmTokenizer.from_pretrained(model_name)
        return model, tokenizer
    
    def get_save_path(self, prefix):
        args = self.args
        if prefix == 'vectorize':
            return f'{config.retr_dir}/vectors_{args.model}.pkl'
        else:
            return f'{config.retr_dir}/topk_{args.model}_{args.metric}.pkl'
    
    def get_protein_vectors(self, proteins):
        args = self.args
        model, tokenizer = self.get_base_model()
        data = self.data_constructor(proteins, tokenizer, device=self.device)
        data_iter = DataLoader(data, batch_size=args.batch_size, collate_fn=data.collate)
        prot2vec = Protein2Vector(model.to(self.device), pooling=args.pooling, hidden_fn=self.hidden_fn)
        vectors = prot2vec(data_iter)
        return vectors
    
    def compute_nns(self, query_vecs, corpus_vecs, k):
        args = self.args
        size = query_vecs.shape[0], corpus_vecs.shape[0], corpus_vecs.shape[1]
        query_vecs = query_vecs.unsqueeze(1).expand(*size)
        corpus_vecs = corpus_vecs.unsqueeze(0).expand(*size)
        data = TensorDataset(query_vecs, corpus_vecs)
        data_iter = DataLoader(data, batch_size=args.batch_size)
        
        scores, indices = [], []
        for query_batch, corpus_batch in tqdm(data_iter, desc='Computing similarities...'):
            query_batch, corpus_batch = query_batch.to(self.device), corpus_batch.to(self.device)
            if args.metric == 'cosine':
                batch_scores = torch.cosine_similarity(query_batch, corpus_batch, -1)
            else:
                batch_scores = pairwise_distance(query_batch, corpus_batch, eps=0.)
            topk, topk_idx = batch_scores.topk(k, 1, largest=args.metric == 'cosine')
            scores.extend(topk.tolist())
            indices.extend(topk_idx.tolist())
        return scores, indices
    
    def __call__(self, proteins=None):
        args = self.args
        if args.mode == 'vectorize':
            vectors = self.get_protein_vectors(list(proteins.values()))
            vectors = dict(names=list(proteins.keys()), vectors=vectors)
            save_path = self.get_save_path(args.mode)
            make_dir(save_path)
            torch.save(vectors, save_path)
        else:
            names, vectors = torch.load(self.get_save_path('vectorize')).values()
            topk, topk_idx = self.compute_nns(vectors, vectors, args.top_k + 1)
            results = {name: dict(tgt_names=[names[i] for i in indices[1:]], scores=scores[1:]) \
                           for name, scores, indices in zip(names, topk, topk_idx)}
            save_path = self.get_save_path(args.mode)
            make_dir(save_path)
            torch.save(results, save_path)
    
