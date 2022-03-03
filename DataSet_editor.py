from KW_extractor import keybert_extractor, Rake_extractor
import pandas as pd
import json
from tqdm.notebook import tqdm
tqdm.pandas()
import re
import  numpy as np


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
    all_movies.to_csv(paths.full_dataset, index=False)

def clean_func(text):
    if text[:11] == "As describe":
        text = ','.join(text.split(',')[1:])
    text = re.sub('\[\d*\]', '', text)
    text = re.sub('\{[\w ]*\}', '', text)
    text = re.sub('[^ a-zA-Z0-9.,\'\!\?\-\$\%\&\(\)_]', '', text)
    text = re.sub('  *', ' ', text)
    text = re.sub(' [.,!?] ', '. ', text)
    text = re.sub(r'http\S+', ' $$', text)
    text = re.sub(r'\S+\.com', ' $$', text)
    text = re.sub(r'ref name[\w\s]+\$\$', '', text)
    text = re.sub(r'\s\$\$', '', text)
    text = re.sub('Plot Synopsis by [\S+\s]+ page', '', text)
    text = re.sub('Plot Synopsis by [\S+\s]+ website', '', text)
    text = re.sub('\[\d*\]', '', text)
    text = re.sub('\{[\w ]*\}', '', text)
    text = re.sub('[^ a-zA-Z0-9.,\'\!\?\-\$\%\&\(\)_]', '', text)
    text = re.sub('  *', ' ', text)
    text = re.sub(' [.,!?] ', '. ', text)
    if text[0] == " ":
        text = text[1:]
    return text

def clean_data(paths):
    all_movies = pd.read_csv(paths.full_dataset)
    all_movies['clean_Plot_old'] = all_movies['clean_Plot']
    all_movies['clean_Plot'] = all_movies['Plot'].apply(clean_func)
    all_movies.to_csv(paths.full_dataset, index=False)

def proccess_genres_func(text):
    if ((not (text)) or (str(text).lower()=="nan")):
        return None
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
    if 'war' in text:
        new_g_list.append('war')
    if 'western' in text:
        new_g_list.append('western')

    # if type(new_g_list)=='NoneType' or len(new_g_list)<1:
    #     return None
    if not(new_g_list):
        # print("+++", new_g_list)
        return None

    new_g = ""
    for g in new_g_list:
        new_g += ", "+g if new_g!="" else g
    return new_g


def proccess_genres(paths):
    all_movies = pd.read_csv(paths.full_dataset)
    all_movies['new_genres'] = all_movies['genres'].apply(proccess_genres_func)
    # print(all_movies['new_genres'].values)
    all_movies.to_csv(paths.full_dataset, index=False)
    print("all_movies shape: ", all_movies.shape)
    all_movies = all_movies[~all_movies.new_genres.isna()]
    print("all_movies shape after dropna(genres): ", all_movies.shape)
    all_movies['num_genres'] = all_movies['new_genres'].apply(lambda x: len(x.split(',')))
    all_movies = all_movies[all_movies['num_genres'] <= 3]
    all_movies.to_csv(paths.filtered_dataset, index=False)


def decide_train_test_sets(args, save_path, filter_len=True):
    df = pd.read_csv(args.data_paths.filtered_dataset)
    print("before dropna: ", df.shape)
    df = df.dropna()
    print("after dropna: ", df.shape)
    if filter_len:
        # filter - keep rows only where plots with len between min_len to max_len
        df['len']=df['clean_Plot'].apply(lambda x: len(x.split('. ')))
        df = df[(df.len>=args.T5.min_len)&(df.len<=args.T5.max_len)]
        # filter - keep rows only where tokenization<512
        from transformers import T5Tokenizer
        tokenizer = T5Tokenizer.from_pretrained(args.T5.model_name)
        df['tok_length'] = df['clean_Plot'].apply(lambda x: len(tokenizer(x)['input_ids']))
        df = df[df['tok_length']<512]
    df['rand_num'] = np.random.rand(df.shape[0])
    df['row_class'] = df.rand_num.apply(lambda x: 'train' if x <= args.dataset.train_prcnt \
        else 'test' if x <= args.dataset.train_prcnt+args.dataset.test_prcnt else 'val')
    df.to_csv(args.data_paths[save_path], index=False)

def kw_extraction(extractor,args,col_name, f_num=1, process_type='sentence_process'):
    """
    adds a column named col_name with the extracted kwywords using keybert
    TODO: decide if to add a type var that will determine what process function to call for in kbextractor
    :param paths:  data paths from main
    :param col_name:  col name to add to the datasets
    :return:
    """
    df = pd.read_csv(args.data_paths[args.T5.run_ds])
    extractor = extractor()
    print(df.shape)
    if process_type=='sentence_process':
        df[col_name] = df['clean_Plot'].progress_apply(extractor.sentence_process, **{'k':f_num})
    elif process_type=='parts_process':
        df[col_name] = df['clean_Plot'].progress_apply(extractor.parts_process, **{'p':f_num})
    df.to_csv(args.data_paths[args.T5.run_ds])

