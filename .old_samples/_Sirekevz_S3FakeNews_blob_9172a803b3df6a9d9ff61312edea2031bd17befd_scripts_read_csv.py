import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from collections import Counter



def return_dataset():

	#load the data
	dataset = pd.read_csv("../data/train.csv")



	# #### PLOT NUMBER OF ARTICLES AND AUTHORS ####
	# sizes = []
	# for batch in dataset.groupby("author"):
	# 	if batch[1].shape[0] < 30:
	# 		sizes.append(batch[1].shape[0])
	# plt.hist(sizes, bins = 100)
	# plt.show()
	# ##############################################



	# convert into numpy array, only text and label
	dataset = np.array(dataset[["text","label"]]) 

	# Remove \n and trailing spaces. Get rid of missing values.
	i = 0
	places_with_nan = []
	print("Removing nans")
	for row in tqdm(dataset):
		if str(row[0]).strip()=="nan" or isinstance(row[0], float):
			places_with_nan.append(i)
			i += 1
			continue

		row[0] = row[0].replace("\n", " ").strip(" ")
		i += 1
	dataset = np.delete(dataset, places_with_nan, axis=0)
	
	data  = []
	print("Removing non letters")
	for line in tqdm(dataset):
		text = line[0]
		text = ''.join(ch.lower() for ch in text if ch.isalpha() or ch == " ")
		data.append([[word for word in text.split(" ") if len(word)>1],line[-1]])

	N = 20761

	# https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-list-of-lists
	flatten = lambda l: [item for sublist in l for item in sublist]

	data_1 = flatten([data[index][0] for index in tqdm(range(N)) if data[index][1]])
	data_0 = flatten([data[index][0] for index in tqdm(range(N)) if not data[index][1]])


	cnt_0 = Counter(data_0)
	cnt_1 = Counter(data_1)

	words_to_be_removed = set()

	print("Loop in all unique words, to select words to be deleted.")
	for unique_word in tqdm(Counter(data_0 + data_1)):
		if cnt_0[unique_word] + cnt_1[unique_word] == 1:
			for row in data:
				if unique_word in row:
					row.remove(unique_word)
					break
				
		if cnt_0[unique_word] + cnt_1[unique_word] < 10:
				words_to_be_removed.add(unique_word)
		elif  0.45 < ((cnt_0[unique_word] / len(data_0)) / 
		((cnt_0[unique_word] / len(data_0))  + (cnt_1[unique_word] / len(data_1)))) < 0.55:
			words_to_be_removed.add(unique_word)

	print("remove selected words")
	for row in tqdm(data):
		row[0] = [word for word in row[0] if word not in words_to_be_removed]
			


	return data










