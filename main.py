import argparse
import yaml
from box import Box
from DataSet_editor import combine_datasets


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='argument parser')
    parser.add_argument("--mode", default='client')
    parser.add_argument('--config', default='config.yaml', type=str,
                        help='Path to YAML config file. Defualt: config.yaml')
    parse_args = parser.parse_args()
    with open(parse_args.config) as f:
        args = Box(yaml.load(f, Loader=yaml.FullLoader))
    combine_datasets(args.data_paths)