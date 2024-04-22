import torch
import random
import numpy as np
import pandas as pd
from copy import deepcopy
from math import ceil
from itertools import chain
from transformers import EsmTokenizer, EsmForMaskedLM
from peft import LoraConfig, get_peft_model, PeftModel
from torch.utils.data import DataLoader
from . import config
from .trainer import RankingTrainer, MetaRankingTrainer
from .dataset.base import MutantSequenceData, RankingSequenceData, MetaRankingSequenceData
from .utils.data import make_dir, split_data
from .utils.score import metrics, group_scores, summarize_scores

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)

def print_trainable_params(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f'Trainable params: {trainable_params} ({100 * trainable_params / all_param:.2f}%)')
    print(f'All params: {all_param}')

class Pipeline():
    def __init__(self, parsed_args, data_constructor=MutantSequenceData,
                 lora_modules=config.lora_modules, score_fn=None):
        if parsed_args.n_sites == [0]:
            parsed_args.n_sites = None
        if not 0 < parsed_args.train_size < 1:
            parsed_args.train_size = int(parsed_args.train_size)
        self.args = parsed_args
        self.device = 'cpu' if parsed_args.force_cpu or not torch.cuda.is_available() else 'cuda'
        self.data_constructor = data_constructor
        self.lora_modules = lora_modules
        self.score_fn = score_fn
        self.get_cv_size = lambda train: 0.75 if len(train['df']) > 50 else 0.5
        set_seed(parsed_args.seed)
    
    def get_base_model(self, load_dir=None):
        args = self.args
        model_name = config.model_dir[args.model]
        if load_dir is None:
            model = EsmForMaskedLM.from_pretrained(model_name)
            for name, param in model.named_parameters():
                if 'contact_head.regression' in name:
                    param.requires_grad = False
        else:
            model = EsmForMaskedLM.from_pretrained(load_dir)
        tokenizer = EsmTokenizer.from_pretrained(model_name)
        return model, tokenizer
        
    def get_save_dir(self, prefix, protein_name, prediction=False):
        args = self.args
        save_dir = '{}/{}/{}/{}/r{}{}{}{}{}{}{}'.format(
            config.pred_dir if prediction else config.ckpt_dir,
            prefix,
            args.model,
            protein_name,
            args.lora_r,
            f'_ts{args.train_size}_cv{args.cross_validation}' if not (args.augment and prefix == 'finetune') else '',
            f'_{args.retr_metric}_mt{args.meta_tasks}' if 'meta' in prefix else '',
            '_' + '-'.join(args.augment) if args.augment else '',
            '_regr' if args.list_size == 1 else '',
            '_ms' if args.n_sites != [1] else '',
            args.save_postfix)
        return save_dir

    def finetune_single(self, train, valid, save_dir=None):
        args = self.args
        if 'transfer' in args.mode:
            load_dir = self.get_save_dir('finetune' if args.mode == 'transfer' else 'meta', train['name'])
            logs = torch.load(load_dir + '/logs.pkl')
            if args.lora_r == 0:
                model, tokenizer = self.get_base_model(load_dir)
            else:
                model, tokenizer = self.get_base_model()
                model = PeftModel.from_pretrained(model, load_dir, is_trainable=True)
            print(f'----------------------Continue training from epoch {logs["best_epoch"]}----------------------')
        else:
            model, tokenizer = self.get_base_model()
            if args.lora_r > 0:
                lora_config = LoraConfig(r=args.lora_r,
                                    lora_alpha=args.lora_r,
                                    target_modules=self.lora_modules,
                                    lora_dropout=0.1,
                                    bias='none')
                model = get_peft_model(model, lora_config)
                print_trainable_params(model)
        
        train_data = RankingSequenceData(train, tokenizer,
                                         mask=args.mask in {'train', 'all'},
                                         list_size=args.list_size,
                                         max_size=args.max_iter * args.train_batch,
                                         constructor=self.data_constructor,
                                         device=self.device)
        train_iter = DataLoader(train_data,
                                batch_size=args.train_batch,
                                shuffle=True,
                                collate_fn=train_data.collate)
        trainer = RankingTrainer(model.to(self.device),
                                 optimizer=args.optimizer,
                                 lr=args.learning_rate,
                                 epochs=args.epochs,
                                 max_grad_norm=args.max_grad_norm,
                                 score_fn=self.score_fn,
                                 eval_metric=args.eval_metric,
                                 log_metrics=metrics,
                                 save_dir=save_dir,
                                 patience=args.patience)
        
        report = {}
        if valid is not None and args.cross_validation > 0:
            eval_data = self.data_constructor(valid, tokenizer,
                                              mask=args.mask in {'eval', 'all'},
                                              device=self.device)
            eval_iter = DataLoader(eval_data,
                                   batch_size=args.eval_batch,
                                   collate_fn=eval_data.collate)
            print('Computing zero-shot scores...')
            _, report['baseline'] = trainer.evaluate_epoch(eval_iter)
        else:
            eval_iter = None
        logs = trainer(train_iter, eval_iter)
        report.update(logs)
        report['best_epoch'] = trainer.best_epoch
        return report
    
    def finetune_single_cv(self, train, test=None):
        args = self.args
        save_dir = self.get_save_dir(args.mode, train['name'])
        if args.cross_validation <= 1:
            report = self.finetune_single(train, test, save_dir)
            torch.save(report, save_dir + '/logs.pkl')
            return report
        
        cv_size = self.get_cv_size(train)
        splits = [split_data(train, cv_size, True) for _ in range(args.cross_validation)]
        epochs = args.epochs
        for i, (cv_train, cv_valid) in enumerate(splits):
            print(f'======================Cross validation: Split {i + 1}======================')
            cv_report = self.finetune_single(cv_train, cv_valid)
            args.epochs = min(args.epochs, len(cv_report[args.eval_metric]))
            if i == 0:
                report = cv_report
                continue
            for key, value in cv_report['baseline'].items():
                report['baseline'][key] += value
            for key in metrics:
                for i in range(args.epochs):
                    report[key][i] += cv_report[key][i]
        
        for key in report['baseline'].keys():
            report['baseline'][key] /= len(splits)
        for key in metrics:
            report[key] = [value / len(splits) for value in report[key][:args.epochs]]
            if key == args.eval_metric: # find best epoch based on cv scores
                best_epoch, best_score = max(enumerate(report[key]), key=lambda x: x[1])
                best_epoch += 1
        print(f'CV-estimated best validating {args.eval_metric} reached at epoch {best_epoch}: {best_score:.3f}')
        print(f'----------------------Training on full data for {best_epoch} epochs----------------------')
        report['best_epoch'] = args.epochs = best_epoch
        logs = self.finetune_single(train, None, save_dir)
        report['train_loss'] = logs['train_loss']
        torch.save(report, save_dir + '/logs.pkl')
        args.epochs = epochs
        return report
    
    def meta_single(self, train, eval_train, eval_test=None):
        args = self.args
        model, tokenizer = self.get_base_model()
        if args.lora_r > 0:
            lora_config = LoraConfig(r=args.lora_r,
                                lora_alpha=args.lora_r,
                                target_modules=self.lora_modules,
                                lora_dropout=0.1,
                                bias='none')
            model = get_peft_model(model, lora_config)
            print_trainable_params(model)
        
        train_splits = [split_data(protein, 0.5, True) for protein in train]
        train_data = MetaRankingSequenceData(train_splits, tokenizer,
                                             adapt_batch_size=args.meta_train_batch,
                                             eval_batch_size=args.meta_eval_batch,
                                             adapt_steps=args.adapt_steps,
                                             mask=args.mask,
                                             list_size=args.list_size,
                                             training=True,
                                             constructor=self.data_constructor,
                                             device=self.device)
        train_iter = DataLoader(train_data,
                                batch_size=args.train_batch,
                                shuffle=True,
                                collate_fn=train_data.collate)
        
        save_dir = self.get_save_dir(args.mode, eval_train['name'])
        trainer = MetaRankingTrainer(model.to(self.device),
                                     optimizer=args.optimizer,
                                     lr=args.learning_rate,
                                     epochs=args.epochs,
                                     max_grad_norm=args.max_grad_norm,
                                     score_fn=self.score_fn,
                                     adapt_lr=args.adapt_lr,
                                     eval_metric=args.eval_metric,
                                     log_metrics=metrics,
                                     save_dir=save_dir,
                                     patience=args.patience)
        
        report = {}
        if args.cross_validation > 0:
            if args.cross_validation == 1:
                eval_splits = [(eval_train, eval_test)]
            else:
                cv_size = self.get_cv_size(eval_train)
                eval_splits = [split_data(eval_train, cv_size, True) for _ in range(args.cross_validation)]
            
            eval_data = MetaRankingSequenceData(eval_splits, tokenizer,
                                                adapt_batch_size=args.meta_train_batch,
                                                eval_batch_size=args.eval_batch,
                                                adapt_steps=args.adapt_steps,
                                                mask=args.mask,
                                                list_size=args.list_size,
                                                training=False,
                                                constructor=self.data_constructor,
                                                device=self.device)
            eval_iter = DataLoader(eval_data,
                                   batch_size=1,
                                   collate_fn=eval_data.collate)
            _, report['baseline'] = trainer.evaluate_epoch(eval_iter)
        else:
            eval_iter = None
        logs = trainer(train_iter, eval_iter)
        report.update(logs)
        report['best_epoch'] = trainer.best_epoch
        torch.save(report, save_dir + '/logs.pkl')
        return report
    
    def test_single(self, train, test):
        args = self.args
        if args.epochs > 0:
            load_dir = self.get_save_dir(args.mode, test['name'])
            if args.lora_r == 0:
                model, tokenizer = self.get_base_model(load_dir)
            else:
                model, tokenizer = self.get_base_model()
                model = PeftModel.from_pretrained(model, load_dir, is_trainable=True)
        else:
            model, tokenizer = self.get_base_model()
        
        test_data = self.data_constructor(test, tokenizer,
                                          mask=args.mask in {'eval', 'all'},
                                          device=self.device)
        test_iter = DataLoader(test_data,
                               batch_size=args.eval_batch,
                               collate_fn=test_data.collate)
        trainer = RankingTrainer(model.to(self.device), log_metrics=[], score_fn=self.score_fn)
        predicts, _ = trainer.evaluate_epoch(test_iter)
        predicts = predicts.tolist()
        
        predicts = pd.Series(predicts, index=test['df'].index, name='prediction')
        report, _ = group_scores(train['df'], predicts, test['df'])
        print('======================Breakdown results======================')
        print(report)
        
        print('Saving model predictions...')
        save_path = self.get_save_dir(args.mode, test['name'], prediction=True)
        save_path += '_base.csv' if args.epochs == 0 else '.csv'
        make_dir(save_path)
        predicts.to_csv(save_path)
        return report
    
    def select_datasets(self, all_proteins):
        args = self.args
        if args.protein in all_proteins.keys():
            return all_proteins[args.protein]
        
        proteins = chain(*all_proteins.values())
        if args.train_size >= 1:
            proteins = filter(lambda x: len(x['df']) > args.train_size, proteins)
        
        if args.protein == 'all':
            return list(proteins)
        if args.protein == 'single-site':
            return list(filter(lambda x: x['n_sites'][-1] == 1, proteins))
        if args.protein == 'multi-site':
            return list(filter(lambda x: x['n_sites'][-1] > 1, proteins))
        if len(args.protein) == 2:
            proteins = list(proteins)
            N, i = int(args.protein[0]), int(args.protein[1])
            n = ceil(len(proteins) / N)
            j = (i - 1) * n
            return proteins[j:j + n]
    
    def get_meta_database(self, all_proteins):
        args = self.args
        database = {name: max(datasets, key=lambda x: len(x['df'])) \
                        for name, datasets in all_proteins.items()}
        topk = torch.load(f'{config.retr_dir}/topk_{args.model}_{args.retr_metric}.pkl')
        return database, topk
    
    def augment_data(self, protein):
        args = self.args
        if args.augment == ['adaptive']:
            aug_models = pd.read_csv(f'{config.retr_dir}/aug_models{args.save_postfix}.csv', index_col=0)
            aug_models = [aug_models.loc[protein['name'], args.train_size]]
        else:
            aug_models = args.augment
        
        raw_data = pd.read_csv(f'{config.raw_data_dir}/{protein["name"]}.csv', index_col='mutant',
                               usecols=aug_models + ['mutant'])
        aug_data = []
        for model_name in aug_models:
            new = deepcopy(protein)
            new['df']['DMS_score'] = raw_data[model_name]
            if new['n_sites'][-1] > 2:
                new, _ = split_data(new, len(new['df']), n_sites=[1, 2])
            aug_data.append(new)
        return aug_data
    
    def __call__(self, all_proteins):
        args = self.args
        proteins = self.select_datasets(all_proteins)
        if args.mode == 'meta':
            database, topk = self.get_meta_database(all_proteins)
        
        reports = {}
        for protein in proteins:
            print(f'**********************Current dataset: {protein["name"]}**********************')
            if protein['name'] == 'CCDB_ECOLI_Tripathi_2016':
                eval_metric = args.eval_metric
                args.eval_metric = 'ndcg' # in case of nan spearmanr
            
            train, test = split_data(protein, args.train_size, n_sites=args.n_sites,
                                     neg_train=args.negative_train, scale=args.list_size == 1)
            if args.test:
                report = self.test_single(train, test)
            elif args.mode != 'meta':
                if args.mode == 'finetune' and args.augment:
                    protein = self.augment_data(protein)[0]
                report = self.finetune_single_cv(train, test)
            else:
                src_name = '_'.join(protein['name'].split('_')[:2])
                tgt_names = topk[src_name]['tgt_names'][:args.meta_tasks]
                meta_train = [database[name] for name in tgt_names]
                if args.augment:
                    meta_train[-len(args.augment):] = self.augment_data(protein)
                if args.meta_tasks < 4:
                    meta_train *= 2
                report = self.meta_single(meta_train, train, test)
            reports[protein['name']] = report
            torch.cuda.empty_cache()
            
            if protein['name'] == 'CCDB_ECOLI_Tripathi_2016':
                args.eval_metric = eval_metric
        
        if args.test and args.protein in {'single-site', 'multi-site', 'all'}:
            save_path = self.get_save_dir(args.mode, args.protein, prediction=True)
            save_path += '_base.pkl' if args.epochs == 0 else '.pkl'
            make_dir(save_path)
            reports = summarize_scores(reports, save_path)
            print('**********************Score summary**********************')
            print(reports[args.eval_metric])
