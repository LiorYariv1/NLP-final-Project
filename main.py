import argparse
import yaml
from box import Box
from DataSet_editor import combine_datasets, kw_extraction, clean_data, proccess_genres, decide_train_test_sets
from KW_extractor import Rake_extractor # , keybert_extractor
from transformers import AutoTokenizer
from T5 import T5_trainer, PlotGenerationModel
from datasets import load_metric
import numpy as np
import wandb
"""
This script will reproduce our model result. The script is using DataSet_editor.py and KW_extractor.py for the data pre-process
and T5.py for training and evaluation
"""

if __name__ =='__main__' :
    parser = argparse.ArgumentParser(description='argument parser')
    parser.add_argument("--mode", default='client')
    parser.add_argument('--config', default='config.yaml', type=str,
                        help='Path to YAML config file. Defualt: config.yaml')
    parse_args = parser.parse_args()
    with open(parse_args.config) as f:
        args = Box(yaml.load(f, Loader=yaml.FullLoader))
    combine_datasets(args.data_paths)
    print("cleaning data")
    clean_data(args.data_paths)
    print("processing genres")
    proccess_genres(args.data_paths)
    print("train test split and final filtering")
    decide_train_test_sets(args,save_path='filtered_dataset_3_30_new',filter_len=True)
    print("extracting keywords. using RAKE model, dividing plots into 3 parts")
    kw_type = 'kw_Rake_p3'
    kw_extraction(Rake_extractor,args,kw_type, 3, process_type='parts_process')
    print("extraction DONE")
    ## to extract keywords using keybert:
    # kw_extraction(keybert_extractor,args.data_paths,kw_type, process_type='parts_process')
    #wandb log init
    wandb.init(project=args.w_and_b.project, name = 'Plot_Generator',
               job_type=kw_type, entity=args.w_and_b.entity,  # ** we added entity, mode
               mode=args.w_and_b.mode, reinit=True)
    print('initializing T5_trainer')
    T5_obj = T5_trainer(args, kw_type)
    print("training...")
    T5_obj.trainer.train()
    print('saving model')
    T5_obj.model.model.save_pretrained(f'{args.T5.model_save_path}__{kw_type}')
    wandb.finish()
    print(f" DONE training , saved to-  {args.T5.model_save_path}__{kw_type}")