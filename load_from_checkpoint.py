# from T5 import T5_trainer
# import argparse
# import yaml
# from box import Box
# import wandb
#
#
# parser = argparse.ArgumentParser(description='argument parser')
# parser.add_argument("--mode", default='client')
# parser.add_argument('--config', default='config_old.yaml', type=str,
#                     help='Path to YAML config file. Defualt: config.yaml')
# parse_args = parser.parse_args()
# with open(parse_args.config) as f:
#     args = Box(yaml.load(f, Loader=yaml.FullLoader))
# # #%%
# kw_type = 'kw_Rake_p3'
# print('initial', args.T5.pretrained_model)
#
# #%%
# checkpoints = { 'epoch_5':'/home/student/project/results__kw_Rake_p3/checkpoint-65700/',
#                'epoch_7':'/home/student/project/results__kw_Rake_p3/checkpoint-91980/',
#                'epoch_8':'/home/student/project/results__kw_Rake_p3/checkpoint-105120/',
#                'epoch_9':'/home/student/project/results__kw_Rake_p3/checkpoint-118260/',
#                 'epoch_10':'/home/student/project/results__kw_Rake_p3/checkpoint-131400/'
# }
# args.T5.pretrained_model = '/home/student/project/model0303__kw_Rake_p3/'
# print(args.T5.pretrained_model)
# args.T5.from_checkpoint = True
# t5_obj = T5_trainer(args, kw_type)
# # df = t5_obj.test_ds.copy()
# for checkpoint_name, checkpoint in checkpoints.items():
#     wandb.init(project=args.w_and_b.project, name = checkpoint_name,
#                job_type=kw_type, entity=args.w_and_b.entity,
#                mode=args.w_and_b.mode, reinit=True)
#     wandb.config.update(args.T5)
#     t5_obj.trainer.train(resume_from_checkpoint=checkpoint)
#     t5_obj.trainer.evaluate(t5_obj.tokenized_datasets['test'])
#     # df[f'plot_{checkpoint_name}'] = t5_obj.repetitions.plots
#     # df.to_csv(args.data_paths.test_results)
#     # prediction_text = [t5_obj.tokenizer.decode()]
#
# # #%%
# args.T5.from_checkpoint = False
# t5_obj = T5_trainer(args, kw_type)
# # df = t5_obj.test_ds.copy()
# for num_beams in [3,5,7,9,10,12]:
#     wandb.init(project=args.w_and_b.project, name = f'full_model_{num_beams}_beams',
#                job_type=kw_type, entity=args.w_and_b.entity,
#                mode=args.w_and_b.mode, reinit=True)
#     t5_obj.change_model_beams(num_beams)
#     print(t5_obj.trainer.model.num_beams)
#     t5_obj.trainer.evaluate(t5_obj.tokenized_datasets['test'])
# #     df[f'full_model_{num_beams}_beams'] = t5_obj.repetitions.plots
# #     df.to_csv('/home/student/project/data/full_model_beams.csv')
# #
# args.T5.pretrained_model = f'/home/student/project/model1902__{kw_type}/'
# t5_obj = T5_trainer(args, kw_type)
# for num_beams in [3,5,7,9,10,12]:
#     wandb.init(project=args.w_and_b.project, name = f'first_model_{kw_type}_beams_{num_beams}',
#                job_type=kw_type, entity=args.w_and_b.entity,
#                mode=args.w_and_b.mode, reinit=True)
#     t5_obj.change_model_beams(num_beams)
#     print(t5_obj.trainer.model.num_beams)
#     t5_obj.trainer.evaluate(t5_obj.tokenized_datasets['test'])

from T5 import T5_trainer
import argparse
import yaml
from box import Box
import wandb
"""
This file implements the comparison between different trained models and different configuration, as showed in the paper.
All results are logged into wandb tool
"""

parser = argparse.ArgumentParser(description='argument parser')
parser.add_argument("--mode", default='client')
parser.add_argument('--config', default='config.yaml', type=str,
                    help='Path to YAML config file. Defualt: config.yaml')
parse_args = parser.parse_args()
with open(parse_args.config) as f:
    args = Box(yaml.load(f, Loader=yaml.FullLoader))
# #%%
kw_type = 'kw_Rake_p3'

"""
This part compares the results of different epochs of the same model. we load from checkpoint, without further training
and evaluate on the test set
"""
"""
This part compares the results of a chosen model between different numbers of beam searches when generating. 
Tomer helped us to understand that this can have an effect on the generated. we load the pretrained model, update the 
num_beams parameter, and evaluate on the test set.
We have compared between 2 models.
"""
# #%%
# args.T5.from_checkpoint = False
# models =  [('full_model','/home/student/project/model0303__kw_Rake_p3/')]
# for model_name, model_path in models:
#     args.T5.pretrained_model = model_path
#     t5_obj = T5_trainer(args, kw_type)
#     df = t5_obj.test_ds.copy()
#     for num_beams in [9,3,5,7,10,12]:
#         wandb.init(project=args.w_and_b.project, name = f'{model_name}_{num_beams}_beams',
#                    job_type=kw_type, entity=args.w_and_b.entity,
#                    mode=args.w_and_b.mode, reinit=True)
#         t5_obj.change_model_beams(num_beams)
#         print(t5_obj.trainer.model.num_beams)
#         t5_obj.trainer.evaluate(t5_obj.tokenized_datasets['test'])
#         if model_name=='full_model':
#             df[f'full_model_{num_beams}_beams'] = t5_obj.repetitions.plots
#             df.to_csv('full_model_beams_new.csv')
# #

checkpoints = { 'epoch_5':'/home/student/project/results__kw_Rake_p3/checkpoint-65700/',
               'epoch_7':'/home/student/project/results__kw_Rake_p3/checkpoint-91980/',
               'epoch_8':'/home/student/project/results__kw_Rake_p3/checkpoint-105120/',
               'epoch_9':'/home/student/project/results__kw_Rake_p3/checkpoint-118260/',
                'epoch_10':'/home/student/project/results__kw_Rake_p3/checkpoint-131400/'
}  #list of model checkpoints to run
args.T5.pretrained_model = '/home/student/project/model0303__kw_Rake_p3/'
print(args.T5.pretrained_model)
args.T5.from_checkpoint = True
t5_obj = T5_trainer(args, kw_type)
df = t5_obj.test_ds.copy()
for checkpoint_name, checkpoint in checkpoints.items():
    wandb.init(project=args.w_and_b.project, name = checkpoint_name,
               job_type=kw_type, entity=args.w_and_b.entity,
               mode=args.w_and_b.mode, reinit=True)
    wandb.config.update(args.T5)
    t5_obj.trainer.train(resume_from_checkpoint=checkpoint)
    t5_obj.trainer.evaluate(t5_obj.tokenized_datasets['test'])
    df[f'plot_{checkpoint_name}'] = t5_obj.repetitions.plots
    df.to_csv('checkpoints_res.csv')

models =  [('first_model',f'/home/student/project/model1902__{kw_type}/')]
for model_name, model_path in models:
    args.T5.pretrained_model = model_path
    t5_obj = T5_trainer(args, kw_type)
    for num_beams in [9,3,5,7,10,12]:
        wandb.init(project=args.w_and_b.project, name = f'{model_name}_{num_beams}_beams',
                   job_type=kw_type, entity=args.w_and_b.entity,
                   mode=args.w_and_b.mode, reinit=True)
        t5_obj.change_model_beams(num_beams)
        print(t5_obj.trainer.model.num_beams)
        t5_obj.trainer.evaluate(t5_obj.tokenized_datasets['test'])
