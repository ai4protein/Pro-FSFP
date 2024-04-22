import argparse
from fsfp import config
from fsfp.utils.data import merge_files

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shuffle', '-s', action='store_true',
                        help='whether to shuffle the examples in each dataset')
    parser.add_argument('--max_length', '-l', type=int, default=1022,
                        help='maximun protein lengh to truncate to')
    parser.add_argument('--save_postfix', '-sp', type=str, default='',
                        help='a custom string to append to the data path')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    data_path = config.data_path.replace('.pkl', f'{args.save_postfix}.pkl')
    datasets = merge_files(config.raw_data_dir, shuffle=args.shuffle, max_len=args.max_length,
                           save_path=data_path)
    