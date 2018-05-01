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
		classes = re.search(r"\<([A-Za-z0-9_; ]+)\>", block)
		if not classes == None:				# Check whether sentences have a class
			classes = classes.group(1)		# Get string containing classes
			all_class = classes.split(" ; ")
			label_list.append(all_class)
			sentence = block.replace(" <"+classes+">", "")
			sentence_list.append(sentence)

	return (sentence_list, label_list)




###################################################### Main Functions ######################################################


# from Resources.time_loc_inferface import get_location, get_time
from Resources import time_loc_inferface as tli

# Nest frame elements under events
def run_extraction1(test_docs, output_file="ext_sent"):
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
			# Reset initial values for new event
			curr_event = {}
			curr_event['type'] = " ".join(list(sentence_events))
			curr_event['location'] = ""
			curr_event['time'] = ""
			curr_event['frames'] = []
			# print("Event : ", curr_event['type'])

			locations = tli.get_location(sentences[i])
			if not locations == None:
				curr_event['location'] = locations
				# print("Location : ", locations)

		if "time" in labels[i]:
			time_attr = tli.get_time(sentences[i])
			if not time_attr == None:
				if len(time_attr) > len(curr_event['time']):
					time_split = time_attr.split(" ")
					for part in time_split:
						if part.startswith("P") or part.startswith("W"):
							time_attr.replace(part, "")
					curr_event['time'] = time_attr.strip()
					# print("Time : ", time_attr)

		sentence_frames = frame_types.intersection(set(labels[i]))
		if len(sentence_frames) > 0:
			# If sentence contains an frame elements
			for frame in sentence_frames:
				if not frame == "time":
					curr_event['frames'].append(frame+" #:# "+sentences[i])
					# print(frame+" #:# "+sentences[i])

	print("-----------------------End of Document-----------------------")

	import json
	with open(output_file+"_extracted", 'w') as fl:
	    json.dump(events_list, fl, indent=4)

# Seperate events and frame sentences
def run_extraction2(test_docs, output_file="ext_sent"):
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
	curr_event['sentence'] = ""
	frames = []

	sentences, labels = get_sent_label_pair(test_docs)
	for i in range(len(sentences)):
		sentence_events = event_types.intersection(set(labels[i]))
		if len(sentence_events) > 0:
			# If sentence contains an event mention
			if not curr_event['type'] == "None":	# for 1st loop
				events_list.append(curr_event)
			# Reset initial values for new event
			curr_event = {}
			curr_event['type'] = " ".join(list(sentence_events))
			curr_event['location'] = ""
			curr_event['time'] = ""
			curr_event['sentence'] = sentences[i]
			# print("Event : ", curr_event['type'])

			locations = tli.get_location(sentences[i])
			if not locations == None:
				curr_event['location'] = locations
				# print("Location : ", locations)

		if "time" in labels[i]:
			time_attr = tli.get_time(sentences[i])
			if not time_attr == None:
				if len(time_attr) > len(curr_event['time']):
					time_split = time_attr.split(" ")
					for part in time_split:
						if part.startswith("P") or part.startswith("W"):
							time_attr.replace(part, "")
					curr_event['time'] = time_attr.strip()
					# print("Time : ", time_attr)

		sentence_frames = frame_types.intersection(set(labels[i]))
		if len(sentence_frames) > 0:
			# If sentence contains an frame elements
			for frame in sentence_frames:
				if not frame == "time":
					frames.append(frame+" #:# "+sentences[i])
					# print(frame+" #:# "+sentences[i])

	print("-----------------------End of Document-----------------------")

	output = {}
	output['Events'] = events_list
	output['Frames'] = frames
	import json
	with open(output_file+"_extracted", 'w') as fl:
	    json.dump(output, fl, indent=4)




###################################################### Calling Functions ######################################################


# test_docs = get_testDoc()
# run_extraction(test_docs)
import os
for filename in os.listdir("./text_data/Data_Docs_Classified"):
	test_docs = get_testDoc(os.path.join("./text_data/Data_Docs_Classified", filename))
	run_extraction2(test_docs, os.path.join("./text_data/Data_Docs_Extracted", filename))
