
import pandas as pd
#%%
import pandas as pd
df = pd.read_csv('/home/student/project/data/filtered_dataset.csv')
#%%
df = df[~df.genres.isna()]
df['genres'] = df.genres.apply(lambda x: x.lower())
#%%
df['len'] = df.Plot.apply(lambda x: len(x.split('. ')))
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
new_model.model.eval()
new_model.model.training
res = new_model(**txt)
#%%
new_model.tokenizer.decode(res[0][0])
#%%
df[df['row_class']=='train'][['Title', 'new_genres', 'kw_Rake_1']+['clean_Plot']][3:4].values
#%%
txt = "<extra_id_0> The Adventures of Dollie </s><extra_id_1> drama </s><extra_id_2>" \
      " outing, river, gypsy, wares, rob, mother, devises, plan, parents, distracted, organized, camp, barrel, camp, escapes, wagon, river, water, barrel, dollie, dollie, parents"
