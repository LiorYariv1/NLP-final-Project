import argparse
import yaml
from box import Box
#%%
parser = argparse.ArgumentParser(description='argument parser')
parser.add_argument("--mode", default='client')
parser.add_argument('--config', default='config.yaml', type=str,
                    help='Path to YAML config file. Defualt: config.yaml')
args = parser.parse_args()
with open(args.config) as f:
    training_args = Box(yaml.load(f, Loader=yaml.FullLoader))
#%%
