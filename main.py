import argparse
import yaml
from box import Box
from DataSet_editor import combine_datasets, kw_extraction, clean_data, proccess_genres, decide_train_test_sets
from KW_extractor import Rake_extractor, keybert_extractor
from transformers import AutoTokenizer
from T5 import T5_trainer, PlotGenerationModel
from datasets import load_metric
import numpy as np
import wandb


if __name__ =='__main__' :
    parser = argparse.ArgumentParser(description='argument parser')
    parser.add_argument("--mode", default='client')
    parser.add_argument('--config', default='config.yaml', type=str,
                        help='Path to YAML config file. Defualt: config.yaml')
    parse_args = parser.parse_args()
    with open(parse_args.config) as f:
        args = Box(yaml.load(f, Loader=yaml.FullLoader))
    # wandb.login()

    # wandb.config.update(args.w_and_b)#FIXME

    # combine_datasets(args.data_paths)

    print("start clean")
    clean_data(args.data_paths)
    print("start done")


    proccess_genres(args.data_paths)

    decide_train_test_sets(args,save_path='filtered_dataset_3_30_new',filter_len=True)

    # print("Start keywords . . . ")
    # print("type: sentence_process, kw_Rake_1_per_sen")
    # kw_extraction(Rake_extractor,args,'kw_Rake_1_per_sen', 1, process_type='sentence_process') ##one-time-run
    # print("1 done")
    print("type: parts_process, kw_Rake_p3")
    kw_extraction(Rake_extractor,args,'kw_Rake_p3', 3, process_type='parts_process') ##one-time-run
    # print("2 done")
    print("Rake DONE")

    ##KEYBERT
    # kw_extraction(keybert_extractor,args.data_paths,'kw_kb_1', k_kw_for_sen) ##one-time-run


    #
    # # T5

    kw_type = 'kw_Rake_p3'
    wandb.init(project=args.w_and_b.project, name = '2702_final',
               job_type=kw_type, entity=args.w_and_b.entity,  # ** we added entity, mode
               mode=args.w_and_b.mode, reinit=True)
    T5_obj = T5_trainer(args, kw_type)
    print("T5_obj Done")
    T5_obj.trainer.train()
    T5_obj.model.model.save_pretrained(f'{args.T5.model_save_path}__{kw_type}')
    wandb.finish()
    print(f"First model DONE -  {args.T5.model_save_path}__{kw_type}")

    # kw_type = 'kw_Rake_1_per_sen'
    # print("Start")
    # wandb.init(project=args.w_and_b.project, name = '2702',
    #            job_type=kw_type, entity=args.w_and_b.entity,  # ** we added entity, mode
    #            mode=args.w_and_b.mode)
    # T5_obj = T5_trainer(args, kw_type)
    # print("T5_obj Done")
    # T5_obj.trainer.train()
    # # T5_obj.model.model.save_pretrained(f'{args.T5.model_save_path}__{kw_type}')
    # # wandb.finish()
    # print(f"Second model DONE -  {args.T5.model_save_path}__{kw_type}")
    # #

    print("OMST<3")

    # model = PlotGenerationModel(args.T5.model_name)

