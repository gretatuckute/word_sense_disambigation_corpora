from utils import *

now = datetime.now()
date_tag = now.strftime("%Y%m%d")

save = True
fname = f'dfs_semcor_all_20211202.pkl'


if __name__ == "__main__":
	df = pd.read_pickle(fname)
	
	# Create a df version with dropped rows according to empty rows in words_no_punc
	df_no_punc = df.drop(df[df.words_no_punc == ''].index)
	
	if save:
		df_no_punc.to_pickle(f'dfs_semcor_all_no_punc_{date_tag}.pkl')
	
	# Obtain all unique words
	vocab = set(df_no_punc.words_no_punc_w_lemma_critical_word.str.lower())
	vocab = {'wedding', 'camp'}
	
	# Get GloVe dict for words of interest
	w2v = read_glove_embed(vocab, GLOVEDIR + 'glove.840B.300d.txt')
	print(f'The following words not available in GloVe dict: {vocab - w2v.keys()}')  # missing words
	
	if save:
		w2v_df = pd.DataFrame.from_dict(w2v, orient='index')
		w2v_df.to_pickle(f'w2v_df_{date_tag}.pkl')
	
	lst_glove_emb = []
	for row in df_no_punc.itertuples():
		word_of_interest = row.words_no_punc_w_lemma_critical_word.lower()
		if word_of_interest in w2v:
			glove_embed = w2v[row.words_no_punc_w_lemma_critical_word.lower()]
		else:
			glove_embed = np.empty(300)
			glove_embed[:] = np.nan
		lst_glove_emb.append(glove_embed)
	
	df_no_punc['glove_emb'] = lst_glove_emb