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
import math
import numpy as np
import matplotlib.pyplot as plt

from sympy import true

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
		#print(filename)
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

def train(positivedata, negativedata):
    posidict = dict(Counter(sum(positivedata,[])))
    # print(len(sum(positivedata,[])))
    # print(len(sum(negativedata,[])))
    negadict = dict(Counter(sum(negativedata,[])))  
    return posidict, negadict


def probilityof(positivetrain, negativetrain, positivedict, negativedict, vocab, category: string, instance, log: bool, laplacesmooth: bool, smoothconstant = 1):

    posinum = len(positivetrain)
    neganum = len(negativetrain)
    suminstance = neganum+posinum   

    if category == "posi" or category == "positive":
        num = posinum
        trainset = positivetrain
        dictuse = positivedict
    else:
        num = neganum
        trainset = negativetrain
        dictuse = negativedict

    plist = []
    p0 = num/suminstance
    plist.append(p0)
    denominator = sum([len(i) for i in trainset])
    # print(sum(dictuse.values())-denominator)
    vsize = len(vocab)

    for i in instance:
        if (not laplacesmooth):
            if i in dictuse:
                probinstance = (dictuse[i]/denominator)
            else: 
                continue
        else: # Smooth
            if i in dictuse:
                probinstance = ((dictuse[i]+smoothconstant)/(denominator+smoothconstant*vsize))
            else: 
                probinstance = ((smoothconstant)/(smoothconstant*vsize))
        plist.append(probinstance)

    if log:
        loglist = list(map(math.log10, plist))
        return sum(loglist)
    
    # if 0 not in plist:
    #     print("prob is not zero")

    return math.prod(plist)

def toCounterDictList(trainingset):
	return [dict(Counter(i)) for i in trainingset]

def accuracy(truePosi, trueNega, falsePosi, falseNega): # Count of all four
	return (truePosi+trueNega)/(truePosi+trueNega+falseNega+falsePosi)

def precision(truePosi, trueNega, falsePosi, falseNega):
	preposi = truePosi/(truePosi+falsePosi)
	prenega = trueNega/(trueNega+falseNega)
	return preposi

def recall(truePosi, trueNega, falsePosi, falseNega):
	recposi = truePosi/(truePosi+falseNega)
	recnega = trueNega/(trueNega+falsePosi)
	return recposi

def fscore(truePosi, trueNega, falsePosi, falseNega, beta: 1):
	pre = precision(truePosi, trueNega, falsePosi, falseNega)
	rec = recall(truePosi, trueNega, falsePosi, falseNega)
	f = (1+beta**2)*((pre*rec)/(pre*(beta**2)+rec))
	return f

def confusionmatrix(truePosi, trueNega, falsePosi, falseNega):
	fig = plt.figure()
	col_labels = ['Predict:+', 'Predict:-']
	row_labels = ['Real:+', 'Real:+']
	table_vals = [[truePosi, falseNega], [trueNega, falsePosi]]
	the_table = plt.table(cellText=table_vals,
                      colWidths=[0.1] * 3,
                      rowLabels=row_labels,
                      colLabels=col_labels,
                      loc='center')
	the_table.auto_set_font_size(False)
	the_table.set_fontsize(24)
	the_table.scale(4, 4)
	plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
	plt.tick_params(axis='y', which='both', right=False, left=False, labelleft=False)

	for pos in ['right','top','bottom','left']:
		plt.gca().spines[pos].set_visible(False)
		
	return 
