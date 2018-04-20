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
for filename in os.listdir("/home/sounak/Resources/Data/rcv1_flat"):
	try:
		run_extraction(os.path.join("/home/sounak/Resources/Data/rcv1_flat", filename), os.path.join("/home/sounak/Resources/Data/rcv1_flat_text", filename))
	except:
		pass
