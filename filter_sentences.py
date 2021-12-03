from utils import *

now = datetime.now()
date_tag = now.strftime("%Y%m%d")

save = True
fname = f'dfs_semcor_all_no_punc_w_glove_emb_20211202.pkl'
fname = f'dfs_semcor_all_20211203.pkl'


if __name__ == "__main__":
	df = pd.read_pickle(fname)
	
	## Obtain stats
	print(f'Number unique words with multiple senses: lemmas {len(df.critical_word_lemma.unique())} with {len(df.sense.unique())} senses')
	
	# Show how many occurrences of senses there are across all sentences
	# omit if there are 800 or more occurrences of a sense --> find the lemma word associated with that sense
	plt.hist(df['num_occurrences_sense_across_all_sents'].values, bins=100)
	plt.show()
	
	# For a given word (lemma) how many unique senses are there?
	plt.hist(df['num_unique_senses_for_lemma'].values, bins=400)
	plt.show()
	
	# What is the max?
	df['num_unique_senses_for_lemma'].max()
	
	# Filter out if there is a lemma with just one unique sense
	
	plt.hist(df.unique(), bins=400)
	plt.xlim([0, 80])
	plt.show()
	










