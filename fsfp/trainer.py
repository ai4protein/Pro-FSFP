import torch
import torch.optim as optim
import torch.nn.functional as F
from peft import PeftModel
from learn2learn.algorithms import MAML
from tqdm import tqdm
from collections import defaultdict
from abc import ABC, abstractmethod
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score
from sklearn.preprocessing import minmax_scale
from .utils.model import pack_lora_layers, replace_modules
from .utils.score import pairwise_ranking_loss, listwise_ranking_loss

def get_optimizer(optimizer, lr, params):
    params = filter(lambda p: p.requires_grad, params)
    if optimizer == 'sgd':
        return optim.SGD(params, lr=lr)
    elif optimizer == 'nag':
        return optim.SGD(params, lr=lr, momentum=0.9, nesterov=True)
    elif optimizer == 'adagrad':
        return optim.Adagrad(params, lr=lr)
    elif optimizer == 'adadelta':
        return optim.Adadelta(params, lr=lr)
    elif optimizer == 'adam':
        return optim.Adam(params, lr=lr)
    else:
        raise ValueError('Unknown optimizer: ' + optimizer)

class TrainerBase(ABC):
    def __init__(self, model, optimizer='adam', lr=1e-4, epochs=100,
                 max_grad_norm=5, lr_decay=None, eval_metric='spearmanr',
                 log_metrics=['spearmanr'], save_dir=None, patience=5, overwrite=True):
        self.model = model
        self.optimizer = get_optimizer(optimizer, lr, model.parameters())
        self.epochs = epochs
        self.max_grad_norm = max_grad_norm
        if lr_decay:
            self.scheduler = optim.lr_scheduler.ExponentialLR(optimizer, lr_decay)
        self.eval_metric = eval_metric
        self.log_metrics = log_metrics
        self.save_dir = save_dir
        self.patience = patience
        self.overwrite = overwrite
        self.curr_epoch = self.curr_iter = self.best_epoch = 0
        self.best_score = float('-inf')
        self.logs = defaultdict(list)
    
    def save_states(self):
        print('Saving model states...')
        save_dir = self.save_dir if self.overwrite else f'{self.save_dir}/epoch_{self.curr_epoch}'
        self.model.save_pretrained(save_dir)
        torch.save(self.logs, self.save_dir + '/logs.pkl')
    
    @abstractmethod
    def predict(self, batch):
        pass
    
    @abstractmethod
    def compute_loss(self, batch):
        pass
    
    def train_step(self, batch): # perform one gradient update
        self.optimizer.zero_grad()
        loss = self.compute_loss(batch)
        loss.backward()
        if self.max_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        return loss.item()
    
    @abstractmethod
    def compute_metrics(self, predicts, targets, labels):
        ''' Return a dict of metrics'''
        pass
    
    def evaluate_epoch(self, eval_iter):
        self.model.eval()
        predicts, targets, labels = [], [], []
        pbar = tqdm(eval_iter, desc='Evaluating')
        
        with torch.no_grad():
            for batch in pbar:
                batch_preds = self.predict(batch)
                # compute metrics on full data
                predicts.append(batch_preds.to('cpu'))
                targets.append(batch['targets'].to('cpu'))
                labels.append(batch['labels'].to('cpu'))
        
        predicts, targets, labels = torch.cat(predicts), torch.cat(targets), torch.cat(labels)
        logs = self.compute_metrics(predicts, targets, labels)
        for key, value in logs.items():
            print('{}: {:.3f}'.format(key, value))
        return predicts, logs
    
    def train_epoch(self, train_iter):
        self.model.train()
        logs = dict(train_loss=0, lr=0)
        pbar = tqdm(train_iter, desc=f'Training epoch {self.curr_epoch + 1}')
        
        for batch in pbar:
            loss = self.train_step(batch)
            self.curr_iter += 1
            logs['train_loss'] += loss
            pbar.set_postfix(loss=loss)
        
        if hasattr(self, 'scheduler'):
            self.scheduler.step()
        logs['train_loss'] /= len(train_iter)
        print('train_loss: {:.3f}'.format(logs['train_loss']))
        logs['lr'] = self.optimizer.param_groups[0]['lr']
        print('lr: {:.1e}'.format(logs['lr']))
        return logs
    
    def __call__(self, train_iter, eval_iter=None):
        for epoch in range(self.epochs):
            logs = self.train_epoch(train_iter)
            for key, value in logs.items():
                self.logs[key].append(value)
            self.curr_epoch += 1
            if eval_iter is None:
                continue
            
            _, logs = self.evaluate_epoch(eval_iter)
            for key, value in logs.items():
                self.logs[key].append(value)
            
            score = logs[self.eval_metric]
            if score > self.best_score:
                self.best_epoch = self.curr_epoch
                self.best_score = score
                if self.save_dir:
                    self.save_states()
            elif self.curr_epoch - self.best_epoch >= self.patience:
                print(f'Early stopped at epoch {self.curr_epoch}')
                print(f'Best validating {self.eval_metric} reached at epoch {self.best_epoch}: {self.best_score:.3f}')
                break
            
        if self.save_dir and eval_iter is None:
            self.save_states()
        return self.logs

class RankingTrainer(TrainerBase):
    def __init__(self, model, margin=1.0, pair_fn='hinge', score_fn=None, **kwargs):
        super().__init__(model, **kwargs)
        self.margin = margin
        self.pair_fn = pair_fn
        self.score_fn = score_fn
    
    def predict(self, batch):
        if self.score_fn is not None:
            return self.score_fn(self.model, batch)
        
        logits = self.model(**batch['sequences']).logits
        log_probs = torch.log_softmax(logits, dim=-1) # batch_size * length * num_aa
        
        predicts = []
        for inv_idx, positions, wt_aas, mt_aas in zip(
                batch['inv_seq_idx'], batch['positions'], batch['wt_aas'], batch['mt_aas']):
            log_prob = log_probs[inv_idx]
            predict = log_prob[positions, mt_aas] - log_prob[positions, wt_aas]
            predicts.append(predict.sum().unsqueeze(0))
        return torch.cat(predicts)
    
    def compute_loss(self, batch):
        predicts = self.predict(batch)        
        predicts, targets = predicts[batch['inv_list_idx']], batch['targets'][batch['inv_list_idx']]
        list_size = batch['inv_list_idx'].shape[1]
        if list_size == 1:
            loss = F.mse_loss(predicts, targets)
        elif list_size == 2:
            loss = pairwise_ranking_loss(predicts[:,0], predicts[:,1], targets[:,0], targets[:,1],
                                         self.pair_fn, self.margin)
        else:
            loss = listwise_ranking_loss(predicts, targets)        
        return loss
    
    def compute_metrics(self, predicts, targets, labels):
        logs = {}
        for metric in self.log_metrics:
            if metric == 'spearmanr':
                logs[metric] = spearmanr(predicts, targets).statistic
            elif metric == 'ndcg':
                std_tgts = minmax_scale(targets.unsqueeze(0), (0, 5), axis=1)
                logs[metric] = ndcg_score(std_tgts, predicts.unsqueeze(0))
            elif metric == 'topk_pr':
                k = min(len(predicts), 30)
                indices = predicts.topk(k).indices
                logs[metric] = torch.count_nonzero(labels[indices]).item() / k
            else:
                raise ValueError('Unknown metric: ' + metric)
        return logs
    
class MetaRankingTrainer(RankingTrainer):
    def __init__(self, model, adapt_lr=1e-4, first_order=True, **kwargs):
        super().__init__(model, **kwargs)
        if isinstance(model, PeftModel):
            self.adapter_name, adapter = pack_lora_layers(model)
            self.adapter = MAML(adapter, adapt_lr, first_order)
        else:
            self.model = MAML(model, adapt_lr, first_order, allow_nograd=True)
    
    def fast_adapt(self, adapt_batch, eval_batch, training=True):
        # copy model for meta-training
        if isinstance(self.model, PeftModel): # replace the adapter with a cloned one
            cloned_adapter = self.adapter.clone()
            replace_modules(self.model, self.adapter_name, cloned_adapter.module)
            adapt = cloned_adapter.adapt
        else: # simply copy the full model
            backup = self.model
            self.model = self.model.clone()
            adapt = self.model.adapt
        
        for batch in adapt_batch:
            adapt_loss = self.compute_loss(batch)
            adapt(adapt_loss)
        
        if training: # compute loss for training, eval_batch should be ranking dataset
            output = self.compute_loss(eval_batch)
        else: # make predictions for testing, eval_batch should be regular dataset
            with torch.no_grad():
                self.model.eval()
                output = self.predict(eval_batch)
                self.model.train()
        
        if isinstance(self.model, PeftModel):
            replace_modules(self.model, self.adapter_name, self.adapter.module)
        else:
            self.model = backup
        return output
    
    def train_step(self, batch):
        loss = []
        self.optimizer.zero_grad()
        for adapt_batch, eval_batch in zip(batch['adapt_batches'], batch['eval_batches']):
            eval_loss = self.fast_adapt(adapt_batch, eval_batch)
            if not eval_loss.isfinite():
                continue
            eval_loss.backward()
            loss.append(eval_loss.item())
        if len(loss) == 0:
            return 0.
        
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.data.mul_(1. / len(loss))
        if self.max_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        return sum(loss) / len(loss)
    
    def adapt_predict(self, batch):
        predicts = []
        for adapt_batch, eval_batch in zip(batch['adapt_batches'], batch['eval_batches']):
            eval_preds = self.fast_adapt(adapt_batch, eval_batch, training=False)
            predicts.append(eval_preds)
        return predicts
    
    def compute_metrics(self, predicts, targets, labels):
        logs = defaultdict(float)
        for batch_preds, batch_tgts, batch_labels in zip(predicts, targets, labels):
            log = super().compute_metrics(batch_preds, batch_tgts, batch_labels)
            for key, value in log.items():
                logs[key] += value
        
        for key in logs.keys():
            logs[key] /= len(predicts)
        return logs
    
    def evaluate_epoch(self, eval_iter):
        self.model.train()
        predicts = []
        logs = defaultdict(float)
        pbar = tqdm(eval_iter, desc='Evaluating')
        
        for batch in pbar:
            batch_preds = self.adapt_predict(batch)
            batch_preds = [preds.to('cpu') for preds in batch_preds]
            predicts.append(batch_preds)
            batch_tgts = [eval_batch['targets'].to('cpu') for eval_batch in batch['eval_batches']]
            batch_labels = [eval_batch['labels'].to('cpu') for eval_batch in batch['eval_batches']]
            
            log = self.compute_metrics(batch_preds, batch_tgts, batch_labels)
            pbar.set_postfix({self.eval_metric: log[self.eval_metric]})
            for key, value in log.items():
                logs[key] += value
        
        predicts = [torch.cat(preds) for preds in zip(*predicts)]
        for key in logs.keys():
            logs[key] /= len(eval_iter)
            print('{}: {:.3f}'.format(key, logs[key]))
        return predicts, logs
               
