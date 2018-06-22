###################################################### Utility Functions ######################################################

def get_testDoc(file_path="./Intermediate_Files/target_doc"):
	docs = []
	with open(file_path, encoding='latin-1') as fl:
		docs.append(fl.read())
	return docs


def get_trainData():
	import pandas as pd

	df = pd.read_csv("./Intermediate_Files/train_sentences.csv", header=None)
	sentences = df[0].tolist()
	labels = df[1].tolist()

	return (sentences, labels)


def get_testSentences(csv_file="./Intermediate_Files/test_sentences.csv"):
	import pandas as pd

	df = pd.read_csv(csv_file, header=None)
	sentences = df[0].tolist()
	labels = df[1].tolist()

	return (sentences, labels)


def get_classTerms(labelsFile="wordListNew.txt"):
	# Gets terms in different lists seperated by '|'
	labels = []
	terms1 = []
	terms2 = []
	with open(labelsFile) as fl:
		for line in fl:
			if line == "#$#\n":
				labels.append(fl.readline().strip())
				tmp = fl.readline().split("|")
				terms1.append(tmp[0].strip())
				terms2.append(tmp[1].strip())

	return (terms1, terms2, labels)


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


def binary_rel(similarity_matrix1, similarity_matrix2, threshold=0.1):
	from copy import deepcopy
	similarity_matrix = deepcopy(similarity_matrix1)
	for i in range(similarity_matrix1.shape[0]):
		for j in range(similarity_matrix1.shape[1]):
			similarity_matrix[i][j] = 1 if (similarity_matrix1[i][j] > threshold and similarity_matrix2[i][j] > threshold) else 0
	return similarity_matrix






###################################################### Main Functions ######################################################


def run_classifier(terms1, terms2, labels, test_docs):
	import numpy as np
	class_terms_matrix1, tfidf1 = tf_idf_fit_transform(terms1)
	class_terms_matrix2, tfidf2 = tf_idf_fit_transform(terms2)

	test_sentences = doc2sentences(test_docs)
	sentence_matrix1 = tfidf1.transform(test_sentences)
	sentence_matrix2 = tfidf2.transform(test_sentences)

	print("Shape of sentence matrix 1 : ", sentence_matrix1.shape)
	print("Shape of sentence matrix 2 : ", sentence_matrix2.shape)

	from sklearn.metrics.pairwise import cosine_similarity
	similarity_matrix1 = cosine_similarity(sentence_matrix1, class_terms_matrix1)
	similarity_matrix2 = cosine_similarity(sentence_matrix2, class_terms_matrix2)
	label_matrix = binary_rel(similarity_matrix1, similarity_matrix2, threshold=0)

	predictions = []
	for i in range(len(test_sentences)):
		predictions.append(tuple([labels[x] for x in range(label_matrix.shape[1]) if label_matrix[i][x]==1]))

	sent_class_list = []
	# for i in range(len(test_sentences)):
	# 	for j in range(len(predictions[i])):
	# 		sent_class_list.append((test_sentences[i][:-1], predictions[i][j]))
	for i in range(len(test_sentences)):
		if len(predictions[i]) > 0:
			sent_class_list.append((test_sentences[i][:-1], predictions[i]))

	return sent_class_list



def run_pipelineClassifier(terms1, terms2, labels, test_docs, output_file_path_list):
	import numpy as np
	class_terms_matrix1, tfidf1 = tf_idf_fit_transform(terms1)
	class_terms_matrix2, tfidf2 = tf_idf_fit_transform(terms2)

	# print("Shape of sentence matrix 1 : ", sentence_matrix1.shape)
	# print("Shape of sentence matrix 2 : ", sentence_matrix2.shape)

	from sklearn.metrics.pairwise import cosine_similarity
	# similarity_matrix1 = cosine_similarity(sentence_matrix1, class_terms_matrix1)
	# similarity_matrix2 = cosine_similarity(sentence_matrix2, class_terms_matrix2)
	# label_matrix = binary_rel(similarity_matrix1, similarity_matrix2, threshold=0)
    #
	# predictions = []
	# for i in range(len(test_sentences)):
	# 	predictions.append(tuple([labels[x] for x in range(label_matrix.shape[1]) if label_matrix[i][x]==1]))

	# sent_class_list = []
	# # for i in range(len(test_sentences)):
	# # 	for j in range(len(predictions[i])):
	# # 		sent_class_list.append((test_sentences[i][:-1], predictions[i][j]))
	# for i in range(len(test_sentences)):
	# 	if len(predictions[i]) > 0:
	# 		sent_class_list.append((test_sentences[i][:-1], predictions[i]))
    #
	# return sent_class_list


	for test_doc, output_file_path in zip(test_docs, output_file_path_list):
		test_sentences = doc2sentences(test_doc)
		sentence_matrix1 = tfidf1.transform(test_sentences)
		sentence_matrix2 = tfidf2.transform(test_sentences)
		similarity_matrix1 = cosine_similarity(sentence_matrix1, class_terms_matrix1)
		similarity_matrix2 = cosine_similarity(sentence_matrix2, class_terms_matrix2)
		label_matrix = binary_rel(similarity_matrix1, similarity_matrix2, threshold=0)

		predictions = []
		for i in range(len(test_sentences)):
			predictions.append(tuple([labels[x] for x in range(label_matrix.shape[1]) if label_matrix[i][x]==1]))

		from lxml import etree
		document = etree.Element('doc')
		doc_tree = etree.ElementTree(document)
		for i in range(len(test_sentences)):
			# curr_pred = [mlb.classes_[x] for x in range(len(predictions[i])) if predictions[i][x]==1]
			curr_pred = predictions[i]
			etree.SubElement(document, "Sent", classes=", ".join(curr_pred)).text = test_sentences[i]
		doc_tree.write(output_file_path)



def run_classifierAccuracy(terms1, terms2, trainLabels, testSentences, testLabels):
	all_labels = ['tsunami', 'heat_wave', 'cold_wave', 'forest_fire', 'limnic_erruptions', \
				'storm', 'avalanches', 'blizzard', 'earthquake', 'floods', 'hurricane', \
				'drought', 'volcano', 'fire', 'cyclone', 'hail_storms', 'land_slide', \
				'epicentre', 'temperature', 'depth', 'speed', 'magnitude', \
				'terrorist_attack', 'suicide_attack', 'normal_bombing', 'shoot_out', \
				'aviation_hazard', 'train_collision', 'industrial_accident', \
				'vehicular_collision', 'surgical_strikes', 'transport_hazards', 'riots', \
				'epidemic', 'famine', 'casualties', 'name', 'participant']
	disaster_labels = ['tsunami', 'heat_wave', 'cold_wave', 'forest_fire', 'limnic_erruptions', \
				'storm', 'avalanches', 'blizzard', 'earthquake', 'floods', 'hurricane', \
				'drought', 'volcano', 'fire', 'cyclone', 'hail_storms', 'land_slide', \
				'epicentre', 'temperature', 'depth', 'speed', 'magnitude', \
				'casualties', 'name', 'participant']
	health_labels = ['epidemic', 'famine', 'casualties', 'name', 'participant']
	conflict_labels = ['terrorist_attack', 'suicide_attack', 'normal_bombing', 'shoot_out', \
				'aviation_hazard', 'train_collision', 'industrial_accident', \
				'vehicular_collision', 'surgical_strikes', 'transport_hazards', 'riots', \
				'casualties', 'name', 'participant']
	dataset_labels = ['tsunami', 'heat_wave', 'cold_wave', 'forest_fire', 'storm', 'avalanches', \
				'blizzard', 'earthquake', 'floods', 'drought', 'volcano', 'fire', 'cyclone', 'hail_storms',  \
				'land_slide', 'epicentre', 'depth', 'speed', 'magnitude', 'terrorist_attack', \
				'suicide_attack', 'normal_bombing', 'shoot_out', 'aviation_hazard', 'train_collision',  \
				'industrial_accident', 'vehicular_collision', 'surgical_strikes', 'transport_hazards', \
				'epidemic', 'famine', 'casualties']
	import numpy as np
	curr_labels = set(dataset_labels)

	trainLabels_temp = trainLabels	;	trainLabels = []
	terms1_temp = terms1	;	terms1 = []
	terms2_temp = terms2	;	terms2 = []
	for ind in range(len(trainLabels_temp)):
		if len(list(set(trainLabels_temp[ind]).intersection(curr_labels))) > 0:
			trainLabels.append(trainLabels_temp[ind])
			terms1.append(terms1_temp[ind])
			terms2.append(terms2_temp[ind])
	del trainLabels_temp
	del terms1_temp
	del terms2_temp

	curr_labels = []
	for l in trainLabels:
		curr_labels.extend(l)
	testLabels = [list(set(l).intersection(set(curr_labels))) for l in testLabels]

	from sklearn.preprocessing import MultiLabelBinarizer
	mlb = MultiLabelBinarizer(classes=curr_labels)
	train_label_matrix = mlb.fit(trainLabels)
	print("Labels : ", mlb.classes_)
	test_label_matrix = mlb.transform(testLabels)
	print("Shape of label matrix : ", test_label_matrix.shape)
	print(test_label_matrix.sum(axis=0))

	class_terms_matrix1, tfidf1 = tf_idf_fit_transform(terms1)
	class_terms_matrix2, tfidf2 = tf_idf_fit_transform(terms2)
	sentence_matrix1 = tfidf1.transform(testSentences)
	sentence_matrix2 = tfidf2.transform(testSentences)
	print("Shape of sentence matrix 1 : ", sentence_matrix1.shape)
	print("Sentence matrix 1 : ", sentence_matrix1)
	print("Shape of sentence matrix 2 : ", sentence_matrix2.shape)
	print("Shape of class terms matrix 1 : ", class_terms_matrix1.shape)
	print("Shape of class terms matrix 2 : ", class_terms_matrix2.shape)

	from sklearn.metrics.pairwise import cosine_similarity
	similarity_matrix1 = cosine_similarity(sentence_matrix1, class_terms_matrix1)
	similarity_matrix2 = cosine_similarity(sentence_matrix2, class_terms_matrix2)
	predictions = binary_rel(similarity_matrix1, similarity_matrix2, threshold=0)

	# print(predictions)
	# print(test_label_matrix)
	print(test_label_matrix.shape, predictions.shape)
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


terms1, terms2, labels = get_classTerms("wordListNew_bengali.txt")
import os

####################################################################################################
# Create train samples for Bootstrap
# sentence_label_list = []
# input_dir = "../4. Bootstrap Classifier/text_data/docs_for_train_text"
# for filename in os.listdir(input_dir):
# 	test_doc = get_testDoc(os.path.join(input_dir, filename))
# 	sentence_label_list.extend(run_classifier(terms1, terms2, labels, test_doc))
#
# # for filename in os.listdir("/home/sounak/Resources/Data/rcv1_flat_text"):
# # 	test_doc = get_testDoc(os.path.join("/home/sounak/Resources/Data/rcv1_flat_text", filename))
# # 	if len(test_doc[0]) > 0:
# # 		sentence_label_list.extend(run_classifier(terms, labels, test_doc))
#
# import csv
# with open('train_sentences.csv', 'w') as fl:
#    writer = csv.writer(fl)
#    for row in sentence_label_list:
# 	   writer.writerow(row)
####################################################################################################


####################################################################################################
# For classifying pre-labelled sentences and get accuracy
csv_file = "./Intermediate_Files/test_sentences_beng.csv"
testSentences, testLabels = get_testSentences(csv_file)
testLabels = [set(label.replace('\'', '').replace(' ', '').split(',')) for label in testLabels]
for lab in testLabels:
	if '' in lab:
		lab.remove('')

temp = labels
labels = []
for lab in temp:
	labels.append([lab])
del temp

trainLabels = labels
run_classifierAccuracy(terms1, terms2, trainLabels, testSentences, testLabels)
####################################################################################################

####################################################################################################
# # For classifying sentences in docs
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
# run_pipelineClassifier(terms1, terms2, labels, test_doc_list, output_file_path_list)
####################################################################################################
