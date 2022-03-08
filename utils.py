from cgi import test
import re
import os
import glob
import random
from nltk.corpus import stopwords
import nltk
import string
from collections import Counter
from pprint import pprint

REPLACE_NO_SPACE = re.compile("[._–;:!`¦\'?,\"()\[\]]") # add one '–'
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
nltk.download('stopwords')  

def preprocess_text(text):
	stop_words = set(stopwords.words('english'))
	text = REPLACE_NO_SPACE.sub("", text)
	text = REPLACE_WITH_SPACE.sub(" ", text)
	text = re.sub(r'\d+', '', text)
	text = text.lower()
	text="".join([i for i in text if i not in string.punctuation])
	words = text.split()
	return [w for w in words if w not in stop_words]

def load_training_set(percentage_positives, percentage_negatives):
	vocab = set() # I returned list later
	positive_instances = []
	negative_instances = []
	for filename in glob.glob('train/pos/*.txt'):
		if random.random() > percentage_positives:
			continue
		with open(os.path.join(os.getcwd(), filename), 'r', encoding="utf-8") as f:
			contents = f.read()
			contents = preprocess_text(contents)
			positive_instances.append(contents)
			vocab = vocab.union(set(contents))
	for filename in glob.glob('train/neg/*.txt'):
		if random.random() > percentage_negatives:
			continue
		with open(os.path.join(os.getcwd(), filename), 'r', encoding="utf-8") as f:
			contents = f.read()
			contents = preprocess_text(contents)
			negative_instances.append(contents)
			vocab = vocab.union(set(contents))
	return positive_instances, negative_instances, list(vocab)

def load_test_set(percentage_positives, percentage_negatives):
	positive_instances = []
	negative_instances = []
	for filename in glob.glob('test/pos/*.txt'):
		print(filename)
		if random.random() > percentage_positives:
			continue
		with open(os.path.join(os.getcwd(), filename), 'r', encoding="utf-8") as f:
			contents = f.read()
			contents = preprocess_text(contents)
			positive_instances.append(contents)
	for filename in glob.glob('test/neg/*.txt'):
		if random.random() > percentage_negatives:
			continue
		with open(os.path.join(os.getcwd(), filename), 'r', encoding="utf-8") as f:
			contents = f.read()
			contents = preprocess_text(contents)
			negative_instances.append(contents)
	return positive_instances, negative_instances