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


def get_testDocs():
	docs = []
	with open("target_doc", encoding='latin-1') as fl:
		docs.append(fl.read())
	return docs


def tokenize(text):
	from nltk import word_tokenize
	from nltk.stem import WordNetLemmatizer
	wnl = WordNetLemmatizer()
	# import re
	# from nltk.corpus import stopwords
	# cachedStopWords = stopwords.words("english")

	min_length = 3
	words = map(lambda word: word.lower(), word_tokenize(text))
	# words = [word for word in words if word not in cachedStopWords]
	# tokens = (list(map(lambda token: wordnet_lemmatizer(token), words)))
	tokens = [wnl.lemmatize(token) for token in words]
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

def run_basic_twitter(text, labels):
	text_matrix, _ = tf_idf_fit_transform(text)

	from sklearn.preprocessing import LabelEncoder
	le = LabelEncoder()
	labels = le.fit_transform(labels)

	from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(text_matrix, labels, test_size=0.2, random_state=42)

	from sklearn.svm import LinearSVC
	clf = LinearSVC(random_state=42)
	clf.fit(X_train, y_train)
	predictions = clf.predict(X_test)

	from sklearn.metrics import f1_score
	print("Classwise F1 Scores: ")
	print(f1_score(y_test, predictions, average=None))
	print("\n\nMicro F1 Scores: ")
	print(f1_score(y_test, predictions, average='micro'))


def run_twitter_training(twitter_text, labels, test_docs):
	twitter_text_matrix, tfidf = tf_idf_fit_transform(twitter_text)

	test_sentences = doc2sentences(test_docs)
	sentence_matrix = tfidf.transform(test_sentences)

	print("Shape of sentence matrix : ", sentence_matrix.shape)
	# print(len(test_sentences))
	# print(test_sentences[0])

	from sklearn.preprocessing import LabelEncoder
	le = LabelEncoder()
	labels = le.fit_transform(labels)
	print("Class with corresponding lables:")
	print(le.classes_)
	print(le.transform(le.classes_))

	X_train, y_train = (twitter_text_matrix, labels)
	X_test = sentence_matrix

	from sklearn.svm import LinearSVC
	clf = LinearSVC(random_state=42)
	clf.fit(X_train, y_train)
	predictions = clf.predict(X_test)

	with open("sent_pred.txt", "w") as fl:
		for i in range(len(test_sentences)):
			fl.write(str(test_sentences[i])+" :\t: "+str(predictions[i])+"\n\n")


###################################################### Calling Functions ######################################################

text, labels = get_twitter()
# run_basic_twitter(text, labels)
test_docs = get_testDocs()
run_twitter_training(text, labels, test_docs)
