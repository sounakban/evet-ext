###################################################### Main Functions ######################################################


# For FIRE Data
def run_extraction_FIRE(input_file, output_file):
	from xml.etree import cElementTree as ET
	root = ET.fromstring(open(input_file).read())
	text = root.find('TEXT').text
	with open(output_file, "w") as fl:
		fl.write(text.replace('\t', ' ').replace('\n', ' '))


# For RCV1 Data
def run_extraction_RCV(input_file, output_file):
	from xml.etree import cElementTree as ET
	currDoc = ""
	tree = ET.parse(input_file)
	file_text = tree.getroot().find('text')
	for para in file_text:
		# currDoc = " ".join( (currDoc, para.text.encode('ascii','ignore')) )
		currDoc = " ".join( (currDoc, para.text) )
	with open(output_file, "w") as fl:
		fl.write(currDoc.replace('\t', ' ').replace('\n', ' '))



###################################################### Calling Functions ######################################################

import os
# for filename in os.listdir("/home/sounak/Resources/Data/rcv1_flat"):
# 	try:
# 		run_extraction_RCV(os.path.join("/home/sounak/Resources/Data/rcv1_flat", filename), os.path.join("/home/sounak/Resources/Data/rcv1_flat_text", filename))
# 	except:
# 		pass


input_dir = "./text_data/natural_disaster"
output_dir = "./text_data/natural_disaster_text"
for filename in os.listdir(input_dir):
	try:
		run_extraction_FIRE(os.path.join(input_dir, filename), os.path.join(output_dir, filename))
	except:
		pass
