###################################################### Utility Functions ######################################################

def get_twitter():
	import pandas as pd
	import preprocessor as p

	with open("2014_india_floods.csv") as fl:
		data = pd.read_csv(fl)

	tweet_id = data['tweet_id'].tolist()
	text = data['tweet_text'].tolist()
	text = [p.clean(t) for t in text]
	labels = data['choose_one_category'].tolist()

	return (text, labels)


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



def get_words_for_cat(twitter_text, labels):
	import numpy as np


	twitter_text_matrix, tfidf = tf_idf_fit_transform(twitter_text)

	from sklearn.preprocessing import LabelEncoder
	le = LabelEncoder()
	labels = le.fit_transform(labels)
	class_names = le.classes_
	# print("Class with corresponding lables:")
	# print(class_names)					#Prints class names
	# print(le.transform(class_names))	#Prints corresponding numeric labels

	# Get document vectors of all documents from each class
	class_doc_dict = {}
	for i in le.transform(class_names):
		doc_indices = [index for index, value in enumerate(list(labels)) if value == i]
		class_doc_dict[class_names[i]] = twitter_text_matrix[doc_indices, :]

	# Calculate avg tfidf accross all docs, for terms in a class
	class_term_dict = {}
	for k in class_doc_dict.keys():
		class_term_dict[k] = class_doc_dict[k].mean(axis=0)

	# Get indices of top words for each class
	term_count = 200
	for k in class_term_dict.keys():
		class_term_dict[k] = np.squeeze(np.asarray(class_term_dict[k]))
		class_term_dict[k] = class_term_dict[k].argsort()[-term_count:][::-1]

	index2vocab_map = {v: k for k, v in tfidf.vocabulary_.items()}
	# Get top terms for each class
	for k in class_term_dict.keys():
		class_term_dict[k] = [index2vocab_map[x] for x in class_term_dict[k]]

	# Print terms
	for k in class_term_dict.keys():
		print("\n\nClass : ", k)
		print("Terms : \n", class_term_dict[k])

	# Write to file
	with open("wordList.txt", "w") as fl:
		for k in class_term_dict.keys():
			fl.write("\n\nClass : "+k)
			fl.write("Terms : \n"+str(class_term_dict[k]))


###################################################### Calling Functions ######################################################

text, labels = get_twitter()
get_words_for_cat(text, labels)
