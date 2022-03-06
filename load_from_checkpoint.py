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
#%%
kw_type = 'kw_Rake_p3'
checkpoints = {'tmp_checkpoint':'/home/student/project/results__kw_Rake_p3/checkpoint-105120/'}
for checkpoint_name, checkpoint in checkpoints.items():
    wandb.init(project=args.w_and_b.project, name = 'checkpoint_name',
               job_type=kw_type, entity=args.w_and_b.entity,
               mode=args.w_and_b.mode, reinit=True)
    t5_obj = T5_trainer(args, kw_type)
    t5_obj.trainer.train(resume_from_checkpoint=checkpoint)
    print(t5_obj.trainer.predict(t5_obj.tokenized_datasets['test'][:5]))
    print('done')


# model_path='/home/student/project/results__kw_Rake_p3/checkpoint-105120/'
# t5_obj = T5_trainer(args, 'kw_Rake_p3')
# train_dataset = t5_obj.tokenized_datasets['train']
#
# model_tmp = PlotGenerationModel('t5-base','t5-base')
# training_args = TrainingArguments(output_dir='fakeres',do_train=False, num_train_epochs=0,per_device_train_batch_size=1)
# #%%
# fake_data = train_dataset[0]
# #%%
# trainer = Trainer(model=model_tmp, args= training_args, train_dataset=fake_data)
#                   #%%
# trainer.train(resume_from_checkpoint=model_path)
#
# trainer.model.generate_plot(tmp)
