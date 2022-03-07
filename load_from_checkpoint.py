from transformers import T5ForConditionalGeneration, AutoModelWithLMHead , TrainingArguments, Trainer, T5Config
from T5 import PlotGenerationModel, T5_trainer
import argparse
import yaml
from box import Box
import wandb


parser = argparse.ArgumentParser(description='argument parser')
parser.add_argument("--mode", default='client')
parser.add_argument('--config', default='config.yaml', type=str,
                    help='Path to YAML config file. Defualt: config.yaml')
parse_args = parser.parse_args()
with open(parse_args.config) as f:
    args = Box(yaml.load(f, Loader=yaml.FullLoader))
# #%%
kw_type = 'kw_Rake_p3'
# checkpoints = { 'epoch_5':'/home/student/project/results__kw_Rake_p3/checkpoint-65700/',
#                'epoch_7':'/home/student/project/results__kw_Rake_p3/checkpoint-91980/',
#                'epoch_8':'/home/student/project/results__kw_Rake_p3/checkpoint-105120/',
#                'epoch_9':'/home/student/project/results__kw_Rake_p3/checkpoint-118260/',
#                 'epoch_10':'/home/student/project/results__kw_Rake_p3/checkpoint-131400/'
# }
# t5_obj = T5_trainer(args, kw_type)
# df = t5_obj.test_ds.copy()
# for checkpoint_name, checkpoint in checkpoints.items():
#     wandb.init(project=args.w_and_b.project, name = checkpoint_name,
#                job_type=kw_type, entity=args.w_and_b.entity,
#                mode=args.w_and_b.mode, reinit=True)
#     t5_obj.trainer.train(resume_from_checkpoint=checkpoint)
#     t5_obj.trainer.evaluate(t5_obj.tokenized_datasets['test'])
#     df[f'plot_{checkpoint_name}'] = t5_obj.repetitions.plots
#     df.to_csv(args.data_paths.test_results)
#     # prediction_text = [t5_obj.tokenizer.decode()]

#%%
t5_obj = T5_trainer(args, kw_type)
df = t5_obj.test_ds.copy()
for num_beams in [3,5,7,9,10,12]:
    wandb.init(project=args.w_and_b.project, name = f'full_model_{num_beams}_beams',
               job_type=kw_type, entity=args.w_and_b.entity,
               mode=args.w_and_b.mode, reinit=True)
    t5_obj.change_model_beams(num_beams)
    print(t5_obj.trainer.model.num_beams)
    t5_obj.trainer.evaluate(t5_obj.tokenized_datasets['test'])
    df[f'full_model_{num_beams}_beams'] = t5_obj.repetitions.plots
    df.to_csv('/home/student/project/data/full_model_beams.csv')