from utils import *

semcor = False
masc = True

now = datetime.now()
date_tag = now.strftime("%Y%m%d")

if __name__ == "__main__":
    
    # Get paths
    if semcor:
        paths = glob.glob("semcor/*.xml")
        save_str = 'semcor'
    if masc:
        paths = glob.glob("masc/*/*/*.xml")
        save_str = 'masc'
    
    dfs = []
    for p in tqdm(paths):
        path = join(os.getcwd(), p)
        # print(path)
        # path = join(os.getcwd(), 'semcor/br-c17.xml')
        df_semcor = obtain_df_amb_words(path)
        dfs.append(df_semcor)

    dfs_all = pd.concat(dfs, axis=0).reset_index()
    
    dfs_all['unique_id'] = dfs_all.file_path.str.split('/').str[-1].str.split('.').str[0] + '.' + dfs_all.sent_num.astype(str)
    dfs_all['critical_word_bool'] = [1 if x != None else 0 for x in dfs_all.sense.values]
    
    print(f'Num unique sentences in {save_str} files: {len(dfs_all["unique_id"].unique())}')
    
    # How many ambiuous words do we have:
    df_words_amb = dfs_all[~dfs_all['sense'].isnull()]
    
    print(f'Number unique words with multiple senses: {len(df_words_amb.words_no_punc.unique())}, lemmas {len(df_words_amb.lemma.unique())} with {len(df_words_amb.sense.unique())} senses')

    u, c = np.unique(df_words_amb.sense, return_counts=True)
    d_count = dict(zip(u,c))
    
    ## Create a column that has the sentence without punctuation and with the ambigious word as the lemma (the column of interest that we would
    # ultimately like to use in our model, i.e. a sentence with no punctuation and the critical word lemmatized)
    dfs_all['words_no_punc_w_lemma_critical_word'] = dfs_all['words_no_punc']
    critical_word_index = dfs_all.loc[dfs_all.critical_word_bool == 1].index
    
    # replace critical word with the lemma
    dfs_all.loc[dfs_all.critical_word_bool == 1, 'words_no_punc_w_lemma_critical_word'] = \
    dfs_all.loc[dfs_all.critical_word_bool == 1].lemma
    
    # Append column that denotes how many unique senses each critical word has
    lst_num_sense_occurrences = []
    lst_crit_lemma_words = []
    lst_num_lemma_across_all_sents = []
    lst_num_unique_senses_for_lemma = []
    
    for s in dfs_all.unique_id.unique(): # for each single sentence
        df_sent = dfs_all.query('unique_id == @s')
        crit_word = df_sent.query('critical_word_bool == 1').lemma.values[0]
        crit_sense = df_sent.query('critical_word_bool == 1').sense.values[0]
        lst_crit_lemma_words.append([crit_word] * len(df_sent))

        # how many times does that particular sense appear across all sentences
        num_occurrences_sense_across_all_sents = d_count[crit_sense]
        lst_num_sense_occurrences.append([num_occurrences_sense_across_all_sents] * len(df_sent))
        
        # how many sentences does the lemma, the crit_word, appear in
        num_lemma_across_all_sents = len(dfs_all.query(f'lemma == "{crit_word}"').unique_id.unique())
        lst_num_lemma_across_all_sents.append([num_lemma_across_all_sents] * len(df_sent))
        
        # how many unique senses does that lemma have
        num_unique_senses_for_lemma = len(dfs_all.query(f'lemma == "{crit_word}"').sense.unique())
        lst_num_unique_senses_for_lemma.append([num_unique_senses_for_lemma] * len(df_sent))
        
    
    dfs_all['critical_word_lemma'] =  [item for sublist in lst_crit_lemma_words for item in sublist]
    dfs_all['num_occurrences_sense_across_all_sents'] = [item for sublist in lst_num_sense_occurrences for item in sublist]
    dfs_all['num_lemma_across_all_sents'] = [item for sublist in lst_num_lemma_across_all_sents for item in sublist]
    dfs_all['num_unique_senses_for_lemma'] = [item for sublist in lst_num_unique_senses_for_lemma for item in sublist]
    
    dfs_all.to_pickle(f'dfs_{save_str}_all_{date_tag}.pkl')
    

    
                      
    
    



