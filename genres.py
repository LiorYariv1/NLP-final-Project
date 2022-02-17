
import pandas as pd
#%%
df = pd.read_csv('/home/student/project/data/full_dataset.csv')
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
df[df['new_genres']==None].shape
#%%
lst = list(set(list(df['new_genres'])))
lst
