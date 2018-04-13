###################################################### Utility Functions ######################################################

def get_testDocs():
	docs = []
	with open("target_doc", encoding='latin-1') as fl:
		docs.append(fl.read())
	return docs


def doc2sentences(docs):
	from nltk import sent_tokenize

	sentences = []
	for doc in docs:
		sent_text = sent_tokenize(doc)
		sentences.extend(sent_text)

	return sentences



###################################################### Main Functions ######################################################


def run_conversion(test_docs):
	test_sentences = doc2sentences(test_docs)

	import csv

	with open("sent.csv", "w") as fl:
		writer = csv.writer(fl, dialect='excel')
		for sent in test_sentences:
			writer.writerow([sent, 9])


###################################################### Calling Functions ######################################################

test_docs = get_testDocs()
run_conversion(test_docs)
