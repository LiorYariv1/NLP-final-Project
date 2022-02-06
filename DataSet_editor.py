
import pandas as pd
import json

def combine_datasets (paths):
    """
    reads all relevant datasets and saves combined data set.
    1. the cmu relevant dataset is split between 2 files: metadata (relvant columns:  WikiID, genres, language)
     and plot_summaries  (relevant columns: wikiID, plot)
    2. the wiki movies dataset
    :param paths: paths to the datasets
    :return: None
    """
    col_names = ['WikiID', 'Freebase movie ID', 'Title', 'Movie release date', 'Movie box office revenue',
                 'Movie runtime',
                 'Movie languages', 'Movie countries', 'genres']
    df_genre = pd.read_csv(paths.cmu_metadata, sep='\t', header=None, names=col_names)
    df_genre['Movie languages'] = df_genre['Movie languages'].apply(lambda x: ', '.join(json.loads(x).values()))
    df_genre = df_genre[df_genre['Movie languages'].str.contains('English')]
    df_genre = df_genre[['WikiID', 'Title', 'genres', 'Movie languages']]
    df_genre['genres'] = df_genre['genres'].apply(lambda x: ', '.join(json.loads(x).values()))
    df_plots = pd.read_csv(paths.cmu_plot_summaries,
        sep='\t', header=None, names=['WikiID', 'Plot'])
    joined = df_genre.set_index('WikiID').join(df_plots.set_index('WikiID'), how='inner', on='WikiID')
    wiki_df = pd.read_csv(paths.wiki)
    wiki_df = wiki_df[wiki_df['Origin/Ethnicity'].isin(['American', 'British', 'Canadian'])]
    wiki_df = wiki_df[~wiki_df.Title.isin(joined.Title.values)][wiki_df.Genre != 'unknown']
    wiki_df = wiki_df.rename(columns={'Genre':'genres'})
    all_movies=  wiki_df[['Title','genres','Plot']].append(joined[['Title','genres','Plot']])
    all_movies.to_csv(paths.full_dataset)
