###################################################### Utility Functions ######################################################

def get_trainData():
	import pandas as pd

	df = pd.read_csv("./Intermediate_Files/train_sentences.csv", header=None)
	sentences = df[0].tolist()
	labels = df[1].tolist()

	return (sentences, labels)


def get_testSentences():
	import pandas as pd

	df = pd.read_csv("./Intermediate_Files/test_sentences.csv", header=None)
	sentences = df[0].tolist()
	labels = df[1].tolist()

	return (sentences, labels)


def get_testDoc(file_path="./Intermediate_Files/target_doc"):
	with open(file_path, encoding='latin-1') as fl:
		doc = fl.read()
	return doc


def doc2sentences(docs):
	from nltk import sent_tokenize

	sentences = []
	for doc in docs:
		sent_text = sent_tokenize(doc)
		sentences.extend(sent_text)

	return sentences


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


def binary_rel(similarity_matrix, threshold=0.0):
	for i in range(similarity_matrix.shape[0]):
		for j in range(similarity_matrix.shape[1]):
			similarity_matrix[i][j] = 0 if similarity_matrix[i][j] <= threshold else 1
	return similarity_matrix






###################################################### Main Functions ######################################################


def run_classifier(sentences, labels, test_doc_list, output_file_path_list):
	import numpy as np

	train_matrix, tfidf = tf_idf_fit_transform(sentences)

	from sklearn.preprocessing import MultiLabelBinarizer
	mlb = MultiLabelBinarizer()
	label_matrix = mlb.fit_transform(labels)

	from sklearn.multiclass import OneVsRestClassifier
	from sklearn.svm import LinearSVC
	estimator = LinearSVC()
	classifier = OneVsRestClassifier(estimator, n_jobs=-1)
	classifier.fit(train_matrix, label_matrix)

	for test_doc, output_file_path in zip(test_doc_list, output_file_path_list):
		test_sentences = doc2sentences([test_doc])
		sentence_matrix = tfidf.transform(test_sentences)
		print("Shape of sentence matrix : ", sentence_matrix.shape)
		predictions = classifier.predict(sentence_matrix)

		from lxml import etree
		document = etree.Element('doc')
		doc_tree = etree.ElementTree(document)
		for i in range(len(test_sentences)):
			curr_pred = [mlb.classes_[x] for x in range(predictions.shape[1]) if predictions[i][x]==1]
			etree.SubElement(document, "Sent", classes=", ".join(curr_pred)).text = test_sentences[i]
		doc_tree.write(output_file_path)


def run_classifierAccuracy(trainSentences, trainLabels, testSentences, testLabels):
	all_labels = ['tsunami', 'heat_wave', 'cold_wave', 'forest_fire', 'limnic_erruptions', \
				'storm', 'avalanches', 'blizzard', 'earthquake', 'floods', 'hurricane', \
				'drought', 'volcano', 'fire', 'cyclone', 'hail_storms', 'land_slide', \
				'intensity', 'epicentre', 'temperature', 'depth', 'speed', 'magnitude', \
				'terrorist_attack', 'suicide_attack', 'normal_bombing', 'shoot_out', \
				'aviation_hazard', 'train_collision', 'industrial_accident', \
				'vehicular_collision', 'surgical_strikes', 'transport_hazards', 'riots', \
				'epidemic', 'famine', 'time', 'place', 'type', 'reason', 'after_effects', \
				'casualties', 'name', 'participant']
	disaster_labels = ['tsunami', 'heat_wave', 'cold_wave', 'forest_fire', 'limnic_erruptions', \
				'storm', 'avalanches', 'blizzard', 'earthquake', 'floods', 'hurricane', \
				'drought', 'volcano', 'fire', 'cyclone', 'hail_storms', 'land_slide', \
				'intensity', 'epicentre', 'temperature', 'depth', 'speed', 'magnitude', \
				'time', 'place', 'type', 'reason', 'after_effects', \
				'casualties', 'name', 'participant']
	health_labels = ['epidemic', 'famine', 'time', 'place', 'type', 'reason', 'after_effects', \
				'casualties', 'name', 'participant']
	conflict_labels = ['terrorist_attack', 'suicide_attack', 'normal_bombing', 'shoot_out', \
				'aviation_hazard', 'train_collision', 'industrial_accident', \
				'vehicular_collision', 'surgical_strikes', 'transport_hazards', 'riots', \
				'time', 'place', 'type', 'reason', 'after_effects', \
				'casualties', 'name', 'participant']
	import numpy as np
	curr_labels = set(all_labels)

	trainLabels = [list(set(l).intersection(curr_labels)) for l in trainLabels]
	curr_labels = []
	for l in trainLabels:
		curr_labels.extend(l)
	curr_labels = set(curr_labels)
	testLabels = [list(set(l).intersection(curr_labels))for l in testLabels]

	from sklearn.preprocessing import MultiLabelBinarizer
	mlb = MultiLabelBinarizer(classes=list(curr_labels))
	train_label_matrix = mlb.fit(trainLabels)
	print("Labels : ", mlb.classes_)
	train_label_matrix = mlb.transform(trainLabels)
	test_label_matrix = mlb.transform(testLabels)
	print("Shape of label matrix : ", test_label_matrix.shape)

	train_matrix, tfidf = tf_idf_fit_transform(trainSentences)
	test_matrix = tfidf.transform(testSentences)
	print("Shape of sentence matrix : ", test_matrix.shape)


	from sklearn.multiclass import OneVsRestClassifier
	from sklearn.svm import LinearSVC
	from sklearn.ensemble import RandomForestClassifier
	estimator = LinearSVC()
	# estimator = RandomForestClassifier(n_estimators=50, max_depth=None, min_samples_split=2, random_state=0, n_jobs = -1)
	classifier = OneVsRestClassifier(estimator, n_jobs=-1)
	classifier.fit(train_matrix, train_label_matrix)
	predictions = classifier.predict(test_matrix)

	from sklearn.metrics import f1_score, precision_score, recall_score
	print("All-Precision", precision_score(test_label_matrix, predictions, average=None))
	print("All-Recall", recall_score(test_label_matrix, predictions, average=None))
	print("All-F1", f1_score(test_label_matrix, predictions, average=None))
	print("Micro-Precision", precision_score(test_label_matrix, predictions, average='micro'))
	print("Micro-Recall", recall_score(test_label_matrix, predictions, average='micro'))
	print("Micro-F1", f1_score(test_label_matrix, predictions, average='micro'))
	print("Macro-Precision", precision_score(test_label_matrix, predictions, average='macro'))
	print("Macro-Recall", recall_score(test_label_matrix, predictions, average='macro'))
	print("Macro-F1", f1_score(test_label_matrix, predictions, average='macro'))



###################################################### Calling Functions ######################################################


trainSentences, trainLabels = get_trainData()
trainLabels = [set(label[1:-1].replace('\'', '').replace(' ', '').split(',')) for label in trainLabels]
for labels in trainLabels:
	if '' in labels:
		labels.remove('')


# For classifying sentences in docs
# test_docs_path = "./text_data/DisasterAnnotatedDocs-English-AUKBC"
# output_dir_path = "./text_data/Pipeline_classified"
# import os
# test_doc_list = []
# output_file_path_list = []
# for filename in os.listdir(test_docs_path):
# 	if not filename.endswith(".xml"):
# 		test_doc = get_testDoc(os.path.join(test_docs_path, filename))
# 		output_file_path = "/".join([output_dir_path, filename])
# 		output_file_path = output_file_path + ".xml"
# 		test_doc_list.append(test_doc)
# 		output_file_path_list.append(output_file_path)
#
# run_classifier(trainSentences, trainLabels, test_doc_list, output_file_path_list)



# For classifying pre-labelled sentences and get accuracy
testSentences, testLabels = get_testSentences()
testLabels = [set(label.replace('\'', '').replace(' ', '').split(',')) for label in testLabels]
for labels in testLabels:
	if '' in labels:
		labels.remove('')

run_classifierAccuracy(trainSentences, trainLabels, testSentences, testLabels)
