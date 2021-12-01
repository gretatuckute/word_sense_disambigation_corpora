import numpy as np
import pandas as pd
import xml.etree.ElementTree as et
from os.path import join
import os
import nltk.data
from string import punctuation
import glob

def obtain_df_amb_words(file_path, min_sent_len=8, max_sent_len=12):
    df = pd.read_xml(file_path)
    
    # Tokenize into sentences
    sent_num = 0
    all_sent_nums = []
    sent_len = 0
    for row in df.itertuples():
        sent_len += 1
        word = row.text
        if word in ['.', '?', '!']:
            all_sent_nums.append(sent_num)
            sent_num += 1
            sent_len = 0
        else:
            all_sent_nums.append(sent_num)
    
    df['sent_num'] = all_sent_nums
    
    # Obtain sentence lengths and strip punctuation
    lst_words_no_punc = []
    lst_sent_len_w_punc = []
    lst_sent_len_no_punc = []
    
    for s in df.sent_num.unique():

        df_sent = df.query('sent_num == @s')
        if df_sent.text.isna().sum() > 0:
            print(f'filled nan in {file_path}')
            df_sent.fillna('-', inplace=True)
            
        sent_len_w_punc = df_sent.text.str.split().apply(len).sum()
        lst_sent_len_w_punc.append([sent_len_w_punc] * (sent_len_w_punc))
        
        # get a version of each sentence without punctuation
        temp = df_sent.text.values
        x = [''.join(c for c in s if c not in punctuation) for s in temp]  # obtain list of only words, omitting punctuation
        lst_words_no_punc.append(x)
        
        # now count by omitting the punctuation
        x_no_punc = [s for s in x if s]
        if len(x_no_punc) == 0:
            lst_sent_len_no_punc.append([0]*len(x))
        else:
            lst_sent_len_no_punc.append([len(x_no_punc)] * len(x))
    
    df['words_no_punc'] = [item for sublist in lst_words_no_punc for item in sublist]
    df['sent_len_w_punc'] = [item for sublist in lst_sent_len_w_punc for item in sublist]
    df['sent_len'] = [item for sublist in lst_sent_len_no_punc for item in sublist]
    
    # Obtain sentences that have ambiguous words
    # count number of ambiguous words in each sentence
    lst_num_amb_words = []
    for s in df.sent_num.unique():
        df_sent = df.query('sent_num == @s')
        lst_num_amb_words.append([(df_sent.sense).count()] * len(df_sent))
    
    df['num_amb_words'] = [item for sublist in lst_num_amb_words for item in sublist]
    
    # Obtain sentences that have ambiguous words
    min_sent_len = min_sent_len - 1
    max_sent_len = max_sent_len + 1
    df_amb = df.query(f'num_amb_words == 1 & sent_len > {min_sent_len} & sent_len < {max_sent_len}')
    
    # Add file path
    df_amb['file_path'] = file_path

    
    return df_amb


if __name__ == "__main__":

    # Get paths
    semcor = glob.glob("semcor/*.xml")
    
    dfs_semcor = []
    for semcor_path in semcor:
        path = join(os.getcwd(), semcor_path)
        # print(path)
        # path = join(os.getcwd(), 'semcor/br-c17.xml')
        df_semcor = obtain_df_amb_words(path)
        dfs_semcor.append(df_semcor)

    dfs_semcor_all = pd.concat(dfs_semcor, axis=0).reset_index()
    
    dfs_semcor_all['unique_id'] = dfs_semcor_all.file_path.str.split('/').str[-1].str.split('.').str[0] + '.' + dfs_semcor_all.sent_num.astype(str)
    dfs_semcor_all['critical_word'] = [1 if x != None else 0 for x in dfs_semcor_all.sense.values]
    
    
    print(f'Num unique sentences in semcor files: {len(dfs_semcor_all["unique_id"].unique())}')
    
    # How many ambiuous words do we have:
    df_words_amb = dfs_semcor_all[~dfs_semcor_all['sense'].isnull()]
    
    print(f'Number unique words with multiple senses: {len(df_words_amb.words_no_punc.unique())}, lemmas {len(df_words_amb.lemma.unique())} with {len(df_words_amb.sense.unique())} senses')

    u, c = np.unique(df_words_amb.sense, return_counts=True)
    d_count = dict(zip(u,c))
    
    
    # Append column that denotes how many unique senses each critical word has
    lst_num_unique_senses_crit_word = []
    for s in dfs_semcor_all.unique_id.unique():
        df_sent = dfs_semcor_all.query('unique_id == @s')
        crit_word = df_sent.query('critical_word == 1').words_no_punc.values[0]
        crit_sense = df_sent.query('critical_word == 1').sense.values[0]
        lst_num_unique_senses_crit_word.append([d_count[crit_sense]] * len(df_sent))
        
    dfs_semcor_all['num_unique_senses_crit_word'] = [item for sublist in lst_num_unique_senses_crit_word for item in sublist]
    
    dfs_semcor_all.to_pickle('dfs_semcor_all.pkl')
    
    # Fix the worst contraction cases
    d_contractions = {'nt':'not',
                      'm':'am',
                      'd':'did',
                      's':'is',
                      're':'are',
                      'ca':'can',
                      'll':'will',
                      've':'have',}
    
    
                      
    
    



