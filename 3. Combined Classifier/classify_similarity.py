###################################################### Utility Functions ######################################################

def get_classTerms():
	labels = []
	terms = []
	with open("wordList.txt") as fl:
		# data = fl.readlines(fl)
		for line in fl:
			if line == "#$#\n":
				labels.append(fl.readline().strip())
				terms.append(fl.readline().strip())

	return (terms, labels)


def get_testDoc(file_path="target_doc"):
	docs = []
	with open(file_path, encoding='latin-1') as fl:
		docs.append(fl.read())
	return docs


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
	tfidf = TfidfVectorizer(tokenizer=tokenize, min_df=1, max_df=1.0, max_features=1000, use_idf=True, sublinear_tf=False);
	tdm = tfidf.fit_transform(docs);
	return (tdm, tfidf)


def doc2sentences(docs):
	from nltk import sent_tokenize

	sentences = []
	for doc in docs:
		sent_text = sent_tokenize(doc)
		sentences.extend(sent_text)

	return sentences


def binary_rel(similarity_matrix, threshold=0.0):
	for i in range(similarity_matrix.shape[0]):
		for j in range(similarity_matrix.shape[1]):
			similarity_matrix[i][j] = 0 if similarity_matrix[i][j] <= threshold else 1
	return similarity_matrix






###################################################### Main Functions ######################################################


def run_similarity(terms, labels, test_docs, output_file="pred_sent"):
	import numpy as np
	class_terms_matrix, tfidf = tf_idf_fit_transform(terms)

	test_sentences = doc2sentences(test_docs)
	sentence_matrix = tfidf.transform(test_sentences)

	print("Shape of sentence matrix : ", sentence_matrix.shape)
	# print("Original order of lables:")
	# print(labels)

	from sklearn.metrics.pairwise import cosine_similarity
	similarity_matrix = cosine_similarity(sentence_matrix, class_terms_matrix)
	similarity_matrix = binary_rel(similarity_matrix)

	with open(output_file+"_classified", "w") as fl:
		for i in range(len(test_sentences)):
			predictions = [labels[x] for x in range(similarity_matrix.shape[1]) if similarity_matrix[i][x]==1]
			predictions = " ; ".join(predictions)
			# fl.write(str(test_sentences[i])+" :\t: "+predictions+"\n\n")
			if len(predictions) > 0:
				fl.write("\n"+str(test_sentences[i])[0:-1]+" <"+predictions+">.\n")
			else:
				fl.write(str(test_sentences[i])+" ")


###################################################### Calling Functions ######################################################


terms, labels = get_classTerms()
# test_docs = get_testDocs()
# run_similarity(terms, labels, test_docs)
import os
for filename in os.listdir("./text_data/Data_Docs"):
	test_doc = get_testDoc(os.path.join("./text_data/Data_Docs", filename))
	run_similarity(terms, labels, test_doc, os.path.join("./text_data/Data_Docs_Classified", filename))
