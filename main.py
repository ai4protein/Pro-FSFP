import argparse
import torch
from fsfp import config
from fsfp.dataset.saprot import SaProtMutantData, saprot_zero_shot
from fsfp.pipeline import Pipeline
from fsfp.utils.score import metrics

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', '-m', type=str, choices=['finetune', 'transfer', 'meta', 'meta-transfer'],
                        default='finetune', help='perform finetuning, meta learning or meta-transfer')
    parser.add_argument('--test', '-t', action='store_true',
                        help='load the trained models from checkpoints and test them')
    parser.add_argument('--model', '-md', type=str, choices=config.model_dir.keys(), required=True,
                        help='name of the foundation model')
    parser.add_argument('--protein', '-p', type=str, default='all',
                        help='name of the target protein')
    parser.add_argument('--train_size', '-ts', type=float, required=True,
                        help='few-shot training set size, can be a float number less than 1 to indicate a proportion')
    parser.add_argument('--train_batch', '-tb', type=int, default=10,
                        help='batch size for training (outer batch size in the case of meta learning)')
    parser.add_argument('--eval_batch', '-eb', type=int, default=1000,
                        help='batch size for evaluation')
    parser.add_argument('--lora_r', '-r', type=int, default=16,
                        help='hyper-parameter r of LORA')
    parser.add_argument('--optimizer', '-o', type=str, choices=['sgd', 'nag', 'adagrad', 'adadelta', 'adam'],
                        default='adam', help='optimizer for training (outer loop optimization in the case of meta learning)')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--epochs', '-e', type=int, default=100,
                        help='maximum training epochs')
    parser.add_argument('--max_grad_norm', '-gn', type=float, default=3,
                        help='maximum gradient norm to clip to')
    parser.add_argument('--mask', '-mk', type=str, choices=['train', 'eval', 'all', 'none'], default='none',
                        help='whether to compute masked 0-shot scores')
    parser.add_argument('--list_size', '-ls', type=int, default=5,
                        help='list size for ranking')
    parser.add_argument('--max_iter', '-mi', type=int, default=10,
                        help='maximum number of iterations per training epoch, useless during meta training')
    parser.add_argument('--eval_metric', '-em', type=str, choices=metrics, default='spearmanr',
                        help='evaluation metric')
    parser.add_argument('--retr_metric', '-rm', type=str, default='cosine',
                        help='similarity metric used for retrieving proteins for meta training')
    parser.add_argument('--augment', '-a', nargs='*', type=str, default=[],
                        help='specify one or more models to use their zero-shot scores for data augmentation')
    parser.add_argument('--meta_tasks', '-mt', type=int, default=3,
                        help='number of tasks used for meta training')
    parser.add_argument('--meta_train_batch', '-mtb', type=int, default=10,
                        help='inner batch size for meta training')
    parser.add_argument('--meta_eval_batch', '-meb', type=int, default=64,
                        help='inner batch size for meta testing')
    parser.add_argument('--adapt_lr', '-alr', type=float, default=5e-3,
                        help='learning rate for inner loop')
    parser.add_argument('--adapt_steps', '-as', type=int, default=4,
                        help='number of iterations for inner loop')
    parser.add_argument('--patience', '-pt', type=int, default=15,
                        help='number of epochs to wait until the validation score improves')
    parser.add_argument('--n_sites', '-ns', nargs='+', type=int, default=[1],
                        help='possible numbers of mutation sites in the training data. \
                              setting to 0 means no constraint')
    parser.add_argument('--negative_train', '-neg', action='store_true',
                        help='whether to constraint the training data to negative examples')
    parser.add_argument('--cross_validation', '-cv', type=int, default=5,
                        help='number of splits for cross validation (shuffle & split) on the training set. \
                              if set to 1, the test set will be used for validation; \
                              if set to 0, no testing or validation will be performed.')
    parser.add_argument('--seed', '-s', type=int, default=666666,
                        help='random seed for training')
    parser.add_argument('--save_postfix', '-sp', type=str, default='',
                        help='a custom string to append to all data paths (data, checkpoints and predictions)')
    parser.add_argument('--force_cpu', '-cpu', action='store_true',
                        help='use cpu for training and evaluation even if gpu is available')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    if args.mode == 'transfer':
        assert args.augment
    path = config.data_path.replace('.pkl', f'{args.save_postfix}.pkl')
    proteins = torch.load(path)
    
    if args.model == 'saprot':
        pipeline = Pipeline(args, data_constructor=SaProtMutantData, score_fn=saprot_zero_shot)
    else:
        pipeline = Pipeline(args)
    pipeline(proteins)
