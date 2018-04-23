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


def get_testDocs():
	docs = []
	with open("target_doc", encoding='latin-1') as fl:
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






###################################################### Main Functions ######################################################


def run_similarity(terms, labels, test_docs):
	import numpy as np
	class_terms_matrix, tfidf = tf_idf_fit_transform(terms)

	test_sentences = doc2sentences(test_docs)
	sentence_matrix = tfidf.transform(test_sentences)

	print("Shape of sentence matrix : ", sentence_matrix.shape)

	print("Original order of lables:")
	print(labels)
	from sklearn.preprocessing import LabelEncoder
	le = LabelEncoder()
	labels = le.fit_transform(labels)
	print("Class with corresponding lables:")
	print(le.classes_)
	print(le.transform(le.classes_))

	from sklearn.metrics.pairwise import cosine_similarity
	similarity_matrix = cosine_similarity(sentence_matrix, class_terms_matrix)
	# print(similarity_matrix[0])
	predictions = np.argmax(similarity_matrix, axis=1)
	# predictions = [(predictions[i], similarity_matrix[i][predictions[i]]) for i in range(len(predictions))]
	predictions = [predictions[i] if similarity_matrix[i][predictions[i]]>0.0 else "Irrelevant" for i in range(len(predictions))]

	# with open("sent_pred.txt", "w") as fl:
	# 	for i in range(len(test_sentences)):
	# 		fl.write(str(test_sentences[i])+" :\t: "+str(predictions[i])+"\n\n")


###################################################### Calling Functions ######################################################

terms, labels = get_classTerms()
test_docs = get_testDocs()
run_similarity(terms, labels, test_docs)
