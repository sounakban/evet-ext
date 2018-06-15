################################################################################
#### creates a csv file containing all sentences and corresponding labels  #####
## from PaLinkA format annotated data. It parses all files given a directory ###
################################################################################

def create_AUKBC_labelled(fl_path):
	from bs4 import BeautifulSoup

	file = open(fl_path).read()

	sentences = []
	labels = []
	soup = BeautifulSoup(file,'html.parser')
	#for each in soup.findAll('p'):
		#sentences.append( ' '.join([tag.text.strip() for tag in each.find_all()]) )
	for each in soup.findAll('p'):
		curr_labels = []
		for tag in each.find_all():
			if tag.name.endswith("-arg"):
				curr_labels.append(tag.name[:-4].lower())
			elif tag.name.endswith("_event"):
				curr_labels.append(tag['type'].lower())
			else:
				continue
		if len(curr_labels) > 0:
			sentences.append( ' '.join([tag.text.strip() for tag in each.find_all()]) )
			labels.append( ', '.join(set(curr_labels)) )

	return sentences, labels


all_sentences = []
all_labels = []
import os
input_dir = "./text_data/beng_docs"
output_file = "./Intermediate_Files/test_sentences_beng.csv"
for filename in os.listdir(input_dir):
	if filename.endswith(".xml"):
		try:
			sentences, labels = create_AUKBC_labelled(os.path.join(input_dir, filename))
			all_sentences.extend(sentences)
			all_labels.extend(labels)
		except:
			pass

sentence_label_list = []
for i in range(len(all_sentences)):
	sentence_label_list.append((all_sentences[i], all_labels[i]))
import csv
with open(output_file, 'w') as fl:
   writer = csv.writer(fl)
   for row in sentence_label_list:
	   writer.writerow(row)
