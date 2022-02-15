import argparse
import yaml
from box import Box
from DataSet_editor import combine_datasets,kw_extraction
from KW_extractor import Rake_extractor
from transformers import AutoTokenizer
from T5 import tokenize_fn, PlotGenerationModel

from datasets import load_metric
import numpy as np


if __name__ =='__main__' :
    parser = argparse.ArgumentParser(description='argument parser')
    parser.add_argument("--mode", default='client')
    parser.add_argument('--config', default='config.yaml', type=str,
                        help='Path to YAML config file. Defualt: config.yaml')
    parse_args = parser.parse_args()
    with open(parse_args.config) as f:
        args = Box(yaml.load(f, Loader=yaml.FullLoader))
    # combine_datasets(args.data_paths)
    kw_extraction(Rake_extractor,args.data_paths,'kw_Rake_1')

    # T5
    model_name = args.T5.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokenized_datasets = datasets.map(tokenize_fn, remove_columns=datasets["train"].column_names)
    tokenized_datasets.set_format('torch')

    model = PlotGenerationModel()


