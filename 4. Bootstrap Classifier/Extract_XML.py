###################################################### Main Functions ######################################################


# For FIRE Data
# def run_extraction(input_file, output_file):
# 	from xml.etree import cElementTree as ET
# 	root = ET.fromstring(open(input_file).read())
# 	text = root.find('text').text
# 	with open(output_file, "w") as fl:
# 		fl.write(text)


# For RCV1 Data
def run_extraction(input_file, output_file):
	from xml.etree import cElementTree as ET
	currDoc = ""
	tree = ET.parse(input_file)
	file_text = tree.getroot().find('text')
	for para in file_text:
		# currDoc = " ".join( (currDoc, para.text.encode('ascii','ignore')) )
		currDoc = " ".join( (currDoc, para.text) )
	with open(output_file, "w") as fl:
		fl.write(currDoc)

###################################################### Calling Functions ######################################################

import os
for filename in os.listdir("./text_data/docs_for_train"):
	run_extraction(os.path.join("./text_data/docs_for_train", filename), os.path.join("./text_data/docs_for_train_text", filename))
