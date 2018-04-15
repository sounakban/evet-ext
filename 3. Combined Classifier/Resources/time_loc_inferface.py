################################################################################
## This module contains functions that take sentences as input and returns time
## and location information.
################################################################################

print(":::::::::::::Loading time and location tagging Libraries::::::::::::::\n")
#Time Tagger libraries
import json
from sutime import SUTime
jar_files = "./Resources/python-sutime-master/jars/"
sutime = SUTime(jars=jar_files, mark_time_ranges=True)

# NER Libraries
from nltk.tokenize import word_tokenize
from nltk.tag import StanfordNERTagger
st = StanfordNERTagger('./Resources/stanford-ner-2018-02-27/classifiers/english.all.3class.distsim.crf.ser.gz',\
						'./Resources/stanford-ner-2018-02-27/stanford-ner.jar', encoding='utf-8')

print("\n:::::::::::::All libraries loaded:::::::::::::\n\n")



def get_time(sentence):
	su_out = json.loads(json.dumps(sutime.parse(sentence), sort_keys=True, indent=4))

	if not len(su_out) == 0:
		time_list = [x['value'] for x in su_out if type(x['value'])==type('')]
		return ", ".join(time_list)
	else:
		return None


def get_location(sentence):

	tags = st.tag(word_tokenize(sentence))
	location_position_pair = [(tags[i][0], i) for i in range(len(tags)) if tags[i][1]=='LOCATION']

	if not len(location_position_pair) == 0:
		i=0
		while i < len(location_position_pair)-1:
			# If consucutive words
			if location_position_pair[i][1] + 1 == location_position_pair[i+1][1]:
				location_position_pair[i+1] = (location_position_pair[i][0]+" "+location_position_pair[i+1][0], location_position_pair[i+1][1])
				del(location_position_pair[i])
			i+=1
		locations = [x[0] for x in location_position_pair]
		return "; ".join(locations)
	else:
		return None
