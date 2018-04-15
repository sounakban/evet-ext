###################################################### Utility Functions ######################################################

def get_testDoc(file_path="pred_sent"):
	docs = []
	with open(file_path, encoding='latin-1') as fl:
		docs.append(fl.read())
	return docs


def get_sent_label_pair(test_docs):
	import re
	sentence_list = []
	label_list = []

	text_blocks = test_docs[0].splitlines()
	for block in text_blocks:
		classes = re.search(r"\<([A-Za-z0-9_, ]+)\>", block)
		if not classes == None:				# Check whether sentences have a class
			classes = classes.group(1)		# Get string containing classes
			all_class = classes.split(", ")
			label_list.append(all_class)
			sentence = block.replace(" <"+classes+">", "")
			sentence_list.append(sentence)

	return (sentence_list, label_list)


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


# from Resources.time_loc_inferface import get_location, get_time
from Resources import time_loc_inferface as tli

def run_extraction(test_docs, output_file="ext_sent"):
	import numpy as np

	event_types = set(["Drought", "Earthquake", "Epidemic", "Hurricane", \
						"Rebellion", "Terrorism", "Tornado", "Tsunami"])
	frame_types = set(["displaced_people_and_evacuations", \
						"donation_needs_or_offers_or_volunteering_services", \
						"infrastructure_and_utilities_damage", "injured_or_dead_people", \
						"missing_trapped_or_found_people", "time"])

	events_list = []
	# Set initial values
	curr_event = {}
	curr_event['type'] = "None"
	curr_event['location'] = ""
	curr_event['time'] = ""
	curr_event['frames'] = []

	sentences, labels = get_sent_label_pair(test_docs)
	for i in range(len(sentences)):
		sentence_events = event_types.intersection(set(labels[i]))
		if len(sentence_events) > 0:
			# If sentence contains an event mention
			if not curr_event['type'] == "None":	# for 1st loop
				events_list.append(curr_event)
			curr_event = {}
			curr_event['type'] = " ".join(list(sentence_events))
			curr_event['location'] = ""
			curr_event['time'] = ""
			curr_event['frames'] = []
			print("Event : ", curr_event['type'])

			locations = tli.get_location(sentences[i])
			if not locations == None:
				curr_event['location'] = locations
				print("Location : ", locations)

		if "time" in labels[i]:
			time_attr = tli.get_time(sentences[i])
			if not time_attr == None:
				if len(time_attr) > len(curr_event['time']):
					curr_event['time'] = time_attr
					print("Time : ", time_attr)

		sentence_frames = frame_types.intersection(set(labels[i]))
		if len(sentence_frames) > 0:
			# If sentence contains an frame elements
			for frame in sentence_frames:
				if not frame == "time":
					# if not 'frames' in curr_event:
					# 	curr_event['frames'] = []
					curr_event['frames'].append(frame+" #:# "+sentences[i])
					print(frame+" #:# "+sentences[i])

	print("-----------------------End of Document-----------------------")










	# class_terms_matrix, tfidf = tf_idf_fit_transform(terms)
	#
	# test_sentences = doc2sentences(test_docs)
	# sentence_matrix = tfidf.transform(test_sentences)
	#
	# print("Shape of sentence matrix : ", sentence_matrix.shape)
	# print("Original order of lables:")
	# print(labels)
	#
	# from sklearn.metrics.pairwise import cosine_similarity
	# similarity_matrix = cosine_similarity(sentence_matrix, class_terms_matrix)
	# similarity_matrix = binary_rel(similarity_matrix)
	#
	# with open(output_file+"_output", "w") as fl:
	# 	for i in range(len(test_sentences)):
	# 		predictions = [labels[x] for x in range(similarity_matrix.shape[1]) if similarity_matrix[i][x]==1]
	# 		predictions = " ; ".join(predictions)
	# 		# fl.write(str(test_sentences[i])+" :\t: "+predictions+"\n\n")
	# 		if len(predictions) > 0:
	# 			fl.write("\n"+str(test_sentences[i])[0:-1]+" <"+predictions+">.\n")
	# 		else:
	# 			fl.write(str(test_sentences[i])+" ")


###################################################### Calling Functions ######################################################


# test_docs = get_testDoc()
# run_extraction(test_docs)
import os
for filename in os.listdir("./text_data/Data_Docs_Classified"):
	test_docs = get_testDoc(os.path.join("./text_data/Data_Docs_Classified", filename))
	run_extraction(test_docs, os.path.join("./text_data/Data_Docs_Extracted", filename))
