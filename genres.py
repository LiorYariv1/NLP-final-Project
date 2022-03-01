
import pandas as pd
#%%
import pandas as pd
df = pd.read_csv('/home/student/project/data/filtered_dataset_3_30.csv')
#%%
df = df[~df.genres.isna()]
df['genres'] = df.genres.apply(lambda x: x.lower())
#%%
df['len'] = df.clean_Plot.apply(lambda x: len(x.split('. ')))
#%%
genres = df['genres'].values
#%%
genres = [g.lower() if isinstance(g,str) else g for g in genres ]
genres = list(set(genres))
#%%
genres_splits = []
for g in genres:
    if isinstance(g,str):
        genres_splits+=g.split(', ')
#%%
genres_splits = list(set(genres_splits))
#%%
genres_df = pd.DataFrame(genres_splits, columns=['genre'])
#%%
genres_df.values
#%%
genres_df[genres_df['genre'].str.contains("-")].values
#%%
# # sport
# genres_df[genres_df['genre'].str.contains("sport")].values
df[df['genres'].str.contains("thrill")]['genres'].values
#%%
genres_df[genres_df.genre.str.contains('thrill')].values
#%%
text = 'science fiction drama supernatural, comedy, war, fantasy, family, horror, holific, action/thriller'
#%%
print(text)
text = text.lower()
text = text.replace("rom-com", "romance, comedy")
text = text.replace("rom com", "romance, comedy")
text = text.replace("sci-fi", "science fiction")
text = text.replace("dramedy", "comedy, drama")
text = text.replace("/", ", ")
text = text.replace("-", ", ")
text = text.replace("romantic", "romance")
new_g_list = []

if 'action' in text:
    new_g_list.append('action')
if 'comedy' in text:
    new_g_list.append('comedy')
if 'crime' in text:
    new_g_list.append('crime')
if 'drama' in text:
    new_g_list.append('drama')
if 'fantasy' in text:
    new_g_list.append('fantasy')
if 'horror' in text:
    new_g_list.append('horror')
if 'mystery' in text:  ##mystery
    new_g_list.append('mystery')
if 'roman' in text:
    new_g_list.append('romance')
if 'science fiction' in text:
    new_g_list.append('science fiction')
if 'sport' in text:
    new_g_list.append('sport')
if 'thriller' in text:
    new_g_list.append('thriller')
if 'war' in text:   ## war, western?
    new_g_list.append('war')

# new_g_list = new_g_list.sort() #unneccesary

new_g = ""
for g in new_g_list:
    new_g += ", " + g if new_g!="" else g

print(text)
print(new_g) try #
#%%
text2 = text.split(', ')
text3 = list(set(text2))
print(text)
print(text2)
print(text3)

#%%
# 'rom-com' -> 'romance, comedy'
# 'rom com' -> 'romance, comedy'
# 'war drama' -> 'war, drama'
# 'sci-fi' -> 'science fiction'
# '/' -> ', '
# '-' -> ', '
# 'romantic' -> 'romance'
# 'sports' -> 'sport'

# #NOT str.contains('drama') -> 'drama' #NOT
# ['drama horror thriller film']

#%%
new_genres = df['new_genres'].values
#%%
# df[df['new_genres']==nan][['Plot','genres','new_genres']]
df[['genres','new_genres']][150:200]
#%%
# df[df['genres']=='western']['genres']
# print(df[df['genres']=='western']['genres'].shape)
print(df[df['new_genres']=='war']['genres'].shape)
# print(df['war' is in df['new_genres']].shape)
#%%
print(df[(df['genres'].str.contains('adventure')) | (df['genres']=='adventure')]['genres'].shape)
# print(df[(df['genres'].str.contains('adventure'))]['genres'].shape)
# print(df[(df['genres']=='music') | (df['genres']=='musical')]['genres'].shape)
# print(df[df['new_genres']=='thriller']['genres'].shape)
#%%
print(df['new_genres'].shape)
print(df['new_genres'].dropna().shape)
#%%
tmp = set(df[df.new_genres.isna()]['genres'].values)
#%%
tmp
#%%
# df[df['genres'].str.contains("thrill")]['genres'].values
df[df.new_genres.str.contains("comedy")]
#%%
df2 = df[~df['new_genres'].isna()]
df2.shape
#%%
lst = set(df2['new_genres'].values)
lst
#%%
df2['try'] = df2['new_genres'].apply(lambda x: len(x.split(',')))
#%%
df2[df2['try']<=3].shape
#%%
newset = set(df2[df2['try']<=3]['new_genres'].values)
#%%
newset
#%%
df2.columns
#%%
df2 = df2[df2['try']<=3]
print(df2.shape)
#%%
df2 = df2[['Title', 'genres', 'Plot', 'kw_kb_1', 'kw_Rake_1', 'row_class', 'clean_Plot', 'new_genres']]
#%%
df2.to_csv('/home/student/project/data/filtered_dataset.csv', index=False)
print("done")
#%%
import numpy as np
df2['rand_num'] = np.random.rand(df2.shape[0])
#%%
df2['row_class'] = df2.rand_num.apply(lambda x: 'train' if x<=0.7 else 'test' if x<=0.85 else 'val')
#%%
df2[['row_class','rand_num']]
#%%
print(df2[df2['row_class']=='train'].shape)
print(df2[df2['row_class']=='test'].shape)
print(df2[df2['row_class']=='val'].shape)
#%%
df2[['clean_Plot', 'kw_kb_1', 'kw_Rake_1', 'new_genres']][1004:1005].values
#%%
# df2.shape()
tmpdf = df2[df2['kw_kb_1'].isna()].shape
#%%
from T5 import T5_trainer, PlotGenerationModel
#%%
new_model = PlotGenerationModel('/home/student/project/model1902/', 't5-base')
#%%
print("oooo")
#%%
txt = "<extra_id_0> Fire King </s> <extra_id_1> drama </s> <extra_id_2> fire, running, kingdom, omri, soup"
#%%
txt = new_model.tokenizer(txt, return_tensors="pt")

#%%
df[df['row_class']=='train'][['Title', 'new_genres', 'kw_Rake_1']+['clean_Plot']][3:4].values
#%%
txt = "<extra_id_0> The Adventures of Dollie </s><extra_id_1> drama </s><extra_id_2>" \
      " outing, river, gypsy, wares, rob, mother, devises, plan, parents, distracted, organized, camp, barrel, camp, escapes, wagon, river, water, barrel, dollie, dollie, parents"

#%%
new_model = PlotGenerationModel('/home/student/project/model1902__kw_Rake_1/', 't5-base')
#%%
txt = "<extra_id_0> Fire King </s> <extra_id_1> drama </s> <extra_id_2> fire, running, kingdom, omri, soup"
txt = new_model.tokenizer(txt, return_tensors="pt")
#%%
txt = "<extra_id_0> The skinni girl </s> <extra_id_1> comedy </s> <extra_id_2> spider, omri, pizza, drink, sleep, bite, radioactive, girl, Jane, ginger, building, web, aunt"
txt = new_model.tokenizer(txt, return_tensors="pt")
#%%
new_model.model.eval()
new_model.model.training
res = new_model(**txt)
#%%
new_model.tokenizer.decode(res[0][0])
#%%
df['len'].describe()
#%%
df[(df['len']>1) & (df['len']<30)].shape
#%%
df[(df['len']>25)].shape
#%%
new_model = PlotGenerationModel('/home/student/project/results__kw_Rake_1/checkpoint-93035/', 't5-base')
#%%
import argparse
from box import Box
import yaml
from transformers import trainer, Trainer, AutoModelWithLMHead
#%%
parser = argparse.ArgumentParser(description='argument parser')
parser.add_argument("--mode", default='client')
parser.add_argument('--config', default='config.yaml', type=str,
                    help='Path to YAML config file. Defualt: config.yaml')
parse_args = parser.parse_args()
with open(parse_args.config) as f:
    args = Box(yaml.load(f, Loader=yaml.FullLoader))
#%%
kw_type = 'kw_Rake_1'
T5_obj = T5_trainer(args, kw_type)
print("T5_obj Done")
#%%
T5_obj.trainer.train('/home/student/project/results__kw_Rake_1/checkpoint-93035/')
#%%
txt = "<extra_id_0> The skinni girl </s> <extra_id_1> comedy </s> <extra_id_2> spider, omri, pizza, drink, sleep, bite, radioactive, girl, Jane, ginger, building, web, aunt"
txt = T5_obj.tokenizer(txt, return_tensors="pt")
#%%
T5_obj.trainer.predict(T5_obj.tokenized_datasets['test'])

# model = AutoModelWithLMHead.from_pretrained('/home/student/project/results__kw_Rake_1/checkpoint-93035')
# model = Trainer.
# model = trainer.
#%%
txt = "a. b. c. d. e. f. g. h. i. j. k."
sentences = txt.split('. ')
n = len(sentences)
tmp_n = (n - 2) // (p - 1)
split = [sentences[0], sentences[1]]
if tmp_n > 0:
    split += ['. '.join(sentences[2 + i * tmp_n:2 + (i + 1) * tmp_n]) for i in range(p - 2)]
split += ['. '.join(sentences[2 + (p - 2) * tmp_n:])]
n_list = [1, 1]
if n <= 15:
    x = int(np.ceil(tmp_n * 0.8))
    n_list += [(x if x > 0 else 1, 1, 1)] * (len(split) - 2)
else:
    x = int(np.floor(tmp_n * 0.3))
    n_list += [(x if x > 0 else 1, 1, 1)] * (len(split) - 2)
#%%
print("nisayon omst <3<3")
#%%
from T5 import T5_trainer, PlotGenerationModel
#%%
txt = '<extra_id_0> Avatar <extra_id_1> science fiction, thriller <extra_id_2> cybernet, sims, sim, leaders'
#%%
txt = '<extra_id_0> Blue balls <extra_id_1> comedy <extra_id_2> bed, kitchen, desk, do, door'
#%%
p3_model = PlotGenerationModel('/home/student/project/model1902__kw_Rake_p3', 't5-base')
#%%
txt = p3_model.tokenizer(txt, return_tensors="pt")
#%%
res=p3_model(**txt)
#%%
p3_model.tokenizer.decode(res[0])
#%%


#%%
import pandas as pd
df = pd.read_csv('/home/student/project/data/filtered_dataset.csv')
df.columns
#%%
from transformers import T5Tokenizer
model_name = 't5-base'
#%%
tokenizer = T5Tokenizer.from_pretrained(model_name)

#%%
beam_outputs = p3_model.model.generate(
    **txt,
    max_length=300,
    num_beams=10,
    no_repeat_ngram_size=3,
    num_return_sequences=10
)
#%%
for i in range(len(beam_outputs)):
    res = p3_model.tokenizer.decode(beam_outputs[i], skip_special_tokens=True)
    print(f' result {i}')
    for sen in res.split('.'):
        print(sen,'.')
#%%
from T5 import PlotGenerationModel,repetitions
#%%
p3_model = PlotGenerationModel('/home/student/project/model1902__kw_Rake_p3', 't5-base')
rep_obj =repetitions()
#%%
## Check results
txts = ['<extra_id_0> Blue balls <extra_id_1> drama <extra_id_2> bed, kitchen, desk, do, door']
for txt in txts:
    txt = p3_model.tokenizer(txt, return_tensors="pt")
    beam_outputs = p3_model.model.generate(
        **txt,
        max_length=300,
        num_beams=10,
        no_repeat_ngram_size=3,
        num_return_sequences=10
    )
    for output in beam_outputs:
        plot = p3_model.tokenizer.decode(output, skip_special_tokens=True)
        print('plot: ')
        plot_print = '. \n'.join(plot.split('.'))
        print(plot_print)
        rep_res = {}
        for n in [1, 2, 3]:
            rep_res[n] = rep_obj.intra_repetitions(n, [plot])['mean']
        print(rep_res)
