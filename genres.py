
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
genres_df[genres_df.genre.str.contains('drama')].values

##TODO: CLEAN DATA:  remove {}, remove "as described", remove numbers [], remove *, \ with 's
