###################################################### Main Functions ######################################################


def run_similarity(input_file, output_file):
	from xml.etree import cElementTree as ET
	root = ET.fromstring(open(input_file).read())
	text = root.find('TEXT').text
	with open(output_file, "w") as fl:
		fl.write(text)




	# with open(output_file+"_classified", "w") as fl:
	# 	for i in range(len(test_sentences)):
	# 		predictions = [labels[x] for x in range(similarity_matrix.shape[1]) if similarity_matrix[i][x]==1]
	# 		predictions = " ; ".join(predictions)
	# 		# fl.write(str(test_sentences[i])+" :\t: "+predictions+"\n\n")
	# 		if len(predictions) > 0:
	# 			fl.write("\n"+str(test_sentences[i])[0:-1]+" <"+predictions+">.\n")
	# 		else:
	# 			fl.write(str(test_sentences[i])+" ")


###################################################### Calling Functions ######################################################

import os
for filename in os.listdir("./text_data/natural_disaster"):
	run_similarity(os.path.join("./text_data/natural_disaster", filename), os.path.join("./text_data/natural_disaster_text", filename))
