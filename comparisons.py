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
This part runs different configuration of the model.generate function (changing in the number of beams). 
The generated plot will be saved to csv in order to read them for reproduction for reproduction of the graphs in the paper. 
We also logged all results to WandB for tracking and visualizing the results. 
Our paper only shows the results obtained from the full model. the first model results can also be computed with this 
script and the analyzing notebook.
"""
# #%%
args.T5.from_checkpoint = False
models =  [('full_model','/home/student/project/model0303__kw_Rake_p3/'),('first_model',f'/home/student/project/model1902__{kw_type}/')]
for model_name, model_path in models:
    args.T5.pretrained_model = model_path
    t5_obj = T5_trainer(args, kw_type)
    df = t5_obj.test_ds.copy()
    for num_beams in [3,5,7,9,10,12]:
        wandb.init(project=args.w_and_b.project, name = f'{model_name}_{num_beams}_beams',
                   job_type=kw_type, entity=args.w_and_b.entity,
                   mode=args.w_and_b.mode, reinit=True)
        t5_obj.change_model_beams(num_beams)
        print(t5_obj.trainer.model.num_beams)
        t5_obj.trainer.evaluate(t5_obj.tokenized_datasets['test'])
        df[f'{model_name}_{num_beams}_beams'] = t5_obj.repetitions.plots
        df.to_csv(f'data/{model_name}_beams.csv')
#
"""
This part compares the results of different epochs of the same model. we load from checkpoint, without further training
and evaluate on the test set. to reproduce, load the checkpoints path of the trained model.
The generated plot will be saved to csv in order to read them for reproduction of the graphs in the paper. 
We also logged all results to WandB for tracking and visualizing the results. 
Our paper only shows the results obtained from the full model. the first model results can also be computed with this 
script and the analyzing notebook.

"""
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
