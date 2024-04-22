import os
import json
import pandas as pd
from collections import namedtuple

with open(f'{os.path.dirname(__file__)}/config.json') as f:
    config = json.load(f)
config = namedtuple('config', config.keys())(**config)

pd.set_option('display.max_columns', 7)
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_colwidth', 20)
pd.set_option('display.width', None)
pd.set_option('display.float_format', '{:.3f}'.format)
