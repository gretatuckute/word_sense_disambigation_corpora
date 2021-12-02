
from utils import *

now = datetime.now()
date_tag = now.strftime("%Y%m%d")

if __name__ == "__main__":

    # Get paths
    semcor = glob.glob("semcor/*.xml")
    
    dfs_semcor = []
    for semcor_path in tqdm(semcor):
        path = join(os.getcwd(), semcor_path)
        # print(path)
        # path = join(os.getcwd(), 'semcor/br-c17.xml')
        df_semcor = obtain_df_amb_words(path)
        dfs_semcor.append(df_semcor)

    dfs_semcor_all = pd.concat(dfs_semcor, axis=0).reset_index()
    
    dfs_semcor_all['unique_id'] = dfs_semcor_all.file_path.str.split('/').str[-1].str.split('.').str[0] + '.' + dfs_semcor_all.sent_num.astype(str)
    dfs_semcor_all['critical_word_bool'] = [1 if x != None else 0 for x in dfs_semcor_all.sense.values]
    
    print(f'Num unique sentences in semcor files: {len(dfs_semcor_all["unique_id"].unique())}')
    
    # How many ambiuous words do we have:
    df_words_amb = dfs_semcor_all[~dfs_semcor_all['sense'].isnull()]
    
    print(f'Number unique words with multiple senses: {len(df_words_amb.words_no_punc.unique())}, lemmas {len(df_words_amb.lemma.unique())} with {len(df_words_amb.sense.unique())} senses')

    u, c = np.unique(df_words_amb.sense, return_counts=True)
    d_count = dict(zip(u,c))
    
    ## Create a column that has the sentence without punctuation and with the ambigious word as the lemma (the column of interest that we would
    # ultimately like to use in our model, i.e. a sentence with no punctuation and the critical word lemmatized)
    dfs_semcor_all['words_no_punc_w_lemma_critical_word'] = dfs_semcor_all['words_no_punc']
    critical_word_index = dfs_semcor_all.loc[dfs_semcor_all.critical_word_bool == 1].index
    
    # replace critical word with the lemma
    dfs_semcor_all.loc[dfs_semcor_all.critical_word_bool == 1, 'words_no_punc_w_lemma_critical_word'] = \
    dfs_semcor_all.loc[dfs_semcor_all.critical_word_bool == 1].lemma
    
    # Append column that denotes how many unique senses each critical word has
    lst_num_unique_senses_crit_word = []
    lst_crit_lemma_words = []
    for s in dfs_semcor_all.unique_id.unique():
        df_sent = dfs_semcor_all.query('unique_id == @s')
        crit_word = df_sent.query('critical_word_bool == 1').lemma.values[0]
        crit_sense = df_sent.query('critical_word_bool == 1').sense.values[0]
        lst_num_unique_senses_crit_word.append([d_count[crit_sense]] * len(df_sent))
        lst_crit_lemma_words.append([crit_word] * len(df_sent))
    
    dfs_semcor_all['critical_word_lemma'] =  [item for sublist in lst_crit_lemma_words for item in sublist]
    dfs_semcor_all['num_unique_senses_crit_word'] = [item for sublist in lst_num_unique_senses_crit_word for item in sublist]
    
    dfs_semcor_all.to_pickle(f'dfs_semcor_all_{date_tag}.pkl')
    

    
                      
    
    



