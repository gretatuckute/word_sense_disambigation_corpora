from utils import *

now = datetime.now()
date_tag = now.strftime("%Y%m%d")

save = True

# save_str = 'masc'
fname_masc = f'dfs_masc_all_no_punc_w_glove_emb_20211203.pkl'
fname_semcor = f'dfs_semcor_all_no_punc_w_glove_emb_20211203.pkl'

# fname = f'dfs_semcor_all_20211203.pkl'

if __name__ == "__main__":
	
	## MASC ##
	df_masc = pd.read_pickle(fname_masc)
	df_masc['corpus'] = 'MASC'
	
	## SEMCOR ##
	df_semcor = pd.read_pickle(fname_semcor)
	df_semcor['corpus'] = 'SEMCOR'
	
	df_all = pd.concat([df_masc, df_semcor])
	
	if save:
		df_all.to_pickle(f'MASC-SEMCOR-merged_full_data_df.pkl')
	
	# Extract df for simply ambiguous words for plotting
	df = df_all[~df_all['sense'].isnull()]
	
	## Obtain stats
	print(f'Number unique words with multiple senses: lemmas {len(df.critical_word_lemma.unique())} with {len(df.sense.unique())} senses')
	print(f'Num unique sentences in both files: {len(df["unique_id"].unique())}')
	

	# Show how many occurrences of senses there are across all sentences
	# omit if there are 800 or more occurrences of a sense --> find the lemma word associated with that sense
	plt.hist(df['num_occurrences_sense_across_all_sents'].values, bins=100, color='cornflowerblue')
	plt.ylabel('Count')
	plt.xlabel('Number of occurrences for each word sense')
	plt.title('Number of occurrences for each word sense')
	if save:
		plt.savefig(f'plots/{date_tag}_num_occurrences_for_each_word_sense.png', dpi=300)
	plt.show()
	
	# For a given word (lemma) how many unique senses are there?
	max_senses = df['num_unique_senses_for_lemma'].max() 	# What is the max?

	plt.hist(df['num_unique_senses_for_lemma'].values, color='cornflowerblue',bins=max_senses)
	plt.ylabel('Count')
	plt.xlabel('Number of senses for "ambiguous words"')
	plt.xticks(np.arange(1, max_senses + 1))
	plt.title('Number of senses for "ambiguous words"')
	if save:
		plt.savefig(f'plots/{date_tag}_num_unique_senses_for_lemma.png', dpi=300)
	plt.show()
	
	# Sentence length
	plt.hist(df['sent_len'].values, color='cornflowerblue')
	plt.ylabel('Count')
	plt.xlabel('Sentence length')
	plt.title('Sentence length')
	if save:
		plt.savefig(f'plots/{date_tag}_sentence_lengths.png', dpi=300)
	plt.show()
	










