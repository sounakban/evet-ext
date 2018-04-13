###################################################### Utility Functions ######################################################

def get_text():
	with open("./Data/Tsunami") as fl:
		data = fl.read()
	return data


def treebank2wordnet_pos(treebank_tag):
	from nltk.corpus import wordnet
	if treebank_tag.startswith('J'):
		return wordnet.ADJ
	elif treebank_tag.startswith('V'):
		return wordnet.VERB
	elif treebank_tag.startswith('N'):
		return wordnet.NOUN
	elif treebank_tag.startswith('R'):
		return wordnet.ADV
	else:
		return wordnet.NOUN


def tokenize(text):
	from nltk.tokenize import RegexpTokenizer
	tokenizer = RegexpTokenizer(r'\w+')
	from nltk import pos_tag
	from nltk.stem import WordNetLemmatizer
	wordnet_lemmatizer = WordNetLemmatizer()
	# import re
	from nltk.corpus import stopwords
	cachedStopWords = stopwords.words("english")

	min_length = 3
	words = map(lambda word: word.lower(), tokenizer.tokenize(text))
	words = [word for word in words if word not in cachedStopWords]
	tags = pos_tag(words)		# tags contains tuple pairs of ('words', 'POS tags')
	# tokens = (list(map(lambda token: wordnet_lemmatizer.lemmatize(token), words)))
	tokens = (list(map(lambda token: wordnet_lemmatizer.lemmatize(token[0], treebank2wordnet_pos(token[1])), tags)))
	filtered_tokens = tokens
	# p = re.compile('[a-zA-Z]+');
	# filtered_tokens = list(filter(lambda token: p.match(token) and len(token)>=min_length, tokens));
	return filtered_tokens


# Perform fit and transform input
def tf_idf_fit_transform(docs):
	from sklearn.feature_extraction.text import TfidfVectorizer
	# tfidf = TfidfVectorizer(tokenizer=tokenize, min_df=1, max_df=1.0, max_features=1000, use_idf=True, sublinear_tf=True);
	tfidf = TfidfVectorizer(tokenizer=tokenize, min_df=1, max_df=1.0, max_features=1000, use_idf=True, sublinear_tf=False)
	tdm = tfidf.fit_transform(docs)
	return (tdm, tfidf)



def get_words_for_cat(text):
	import numpy as np

	from nltk.tokenize import sent_tokenize
	text_sentences = sent_tokenize(text)

	text_matrix, tfidf = tf_idf_fit_transform(text_sentences)
	print("done tfidf")

	text_matrix = text_matrix.mean(axis=0)
	print("done mean")


	# Get indices of top words for each class
	term_count = 200
	text_matrix = np.squeeze(np.asarray(text_matrix))	# Make array 1 dimensional
	print("done squeeze")

	text_matrix = text_matrix.argsort()[-term_count:][::-1]
	print("done selection")

	index2vocab_map = {v: k for k, v in tfidf.vocabulary_.items()}
	# Get top terms for each class
	text_matrix = [index2vocab_map[x] for x in text_matrix]

	# Print terms
	print("Terms : \n", " ".join(text_matrix))

	# Write to file
	# with open("wordList.txt", "w") as fl:
	# 	fl.write("Terms : \n"+str(class_term_dict[k]))


###################################################### Calling Functions ######################################################

text = get_text()
get_words_for_cat(text)
