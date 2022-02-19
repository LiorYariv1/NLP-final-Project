import argparse
import yaml
from box import Box
from DataSet_editor import combine_datasets, kw_extraction, clean_data, proccess_genres, decide_train_test_sets
from KW_extractor import Rake_extractor, keybert_extractor
from transformers import AutoTokenizer
from T5 import T5_trainer, PlotGenerationModel
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

    # clean_data(args.data_paths)

    # proccess_genres(args.data_paths)

    # print("Start keywords . . . ")
    # kw_extraction(Rake_extractor,args.data_paths,'kw_Rake_1') ##one-time-run
    # print("Rake DONE")
    # kw_extraction(keybert_extractor,args.data_paths,'kw_kb_1') ##one-time-run

    # decide_train_test_sets(args)

    # # T5
    T5_obj = T5_trainer(args)
    print("T5_obj Done")
    T5_obj.trainer.train()
    T5_obj.model.save_pretrained(args.T5.model_save_path)
    # # input_cols = ['Title', 'genres', 'Plot']
   # # T5_obj.organize_dataset(input_cols)
    print("OMST<3")

    # model = PlotGenerationModel(args.T5.model_name)

