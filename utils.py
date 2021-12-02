import numpy as np
import pandas as pd
from os.path import join
import os
from string import punctuation
import glob
from tqdm import tqdm
from datetime import datetime

GLOVEDIR = '/Users/gt/Dropbox (MIT)/SemComp/features/database/glove_database/'


# Fix the worst contraction cases
# d_contractions = {'nt':'not',
#                   'm':'am',
#                   'd':'did',
#                   's':'is', # not always true
#                   're':'are',
#                   'ca':'can',
#                   'll':'will',
#                   've':'have',}

### OBTAINING SENTENCES FORM XML ###
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
		x = [''.join(c for c in s if c not in punctuation) for s in
			 temp]  # obtain list of only words, omitting punctuation
		lst_words_no_punc.append(x)
		
		# now count by omitting the punctuation
		x_no_punc = [s for s in x if s]
		if len(x_no_punc) == 0:
			lst_sent_len_no_punc.append([0] * len(x))
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


### RSA / GloVe Functions ###
def get_vocab(f1g):
	"""
	Return set of unique tokens in input (assume input is a list of token lists)
	"""
	vocab = set()
	for sent in f1g:
		for token in sent:
			vocab.add(token)
	return vocab


def read_glove_embed(vocab, glove_path):
	"""
	Read through the embedding file to find embeddings for target words in vocab
	Return dict mapping word to embedding (numpy array)
	"""
	w2v = {}
	with open(glove_path, 'r') as f:
		for line in f:
			tokens = line.strip().split(' ')
			w = tokens[0]
			if w in vocab:
				v = tokens[1:]
				w2v[w] = np.array(v, dtype=float)
	return w2v


def get_glove_embed(lst, w2v):
	"""
	lst: list containing words/sentences
	w2v: dict of embeddings

	Fetches glove embeddings for a list of strings. If multi-word: averages the embedding,
	if single-word, just fetches the single embedding.
	"""
	glove_embed = []
	NA_words = set()
	contains_na = []
	n_words = 0
	for sent_idx, sent in enumerate(lst):
		sent_embed = []
		flag_na = False
		if sent_idx % 200 == 0:
			print(sent_idx, flush=True)
		
		# find multi-word words
		split_sent = sent.split(' ')  # contains words
		print(f'Sentence: {split_sent}')
		
		for token_idx, token in enumerate(split_sent):
			print(f'Tokens: {token}')
			if token in w2v:
				sent_embed.append(w2v[token])
			else:
				NA_words.add(token)
				flag_na = True
				a = np.empty((300))
				a[:] = np.nan
				sent_embed.append(a)
				print(f'Added NaN emb for {token}')
		
		#         print(f'Sentence embedding: {sent_embed}')
		glove_embed.append(np.nanmean(np.array(sent_embed), axis=0))
		contains_na.append(flag_na)
	
	try:
		print(f'Number of words with NA glove embedding: {len(NA_words)}')
		print('Example NA words:', list(NA_words))
	except:
		print('No NaN words!')
	
	return glove_embed, contains_na, NA_words