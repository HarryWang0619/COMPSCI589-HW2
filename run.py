from utils import *
from pprint import pprint

def naive_bayes():
	percentage_positive_instances_train = 0.5
	percentage_negative_instances_train = 0.5

	percentage_positive_instances_test  = 0.3
	percentage_negative_instances_test  = 0.3
	
	(pos_train, neg_train, vocab) = load_training_set(percentage_positive_instances_train, percentage_negative_instances_train)
	(pos_test,  neg_test)         = load_test_set(percentage_positive_instances_test, percentage_negative_instances_test)

	print("Number of positive training instances:", len(pos_train))
	print("Number of negative training instances:", len(neg_train))
	print("Number of positive test instances:", len(pos_test))
	print("Number of negative test instances:", len(neg_test))

	with open('vocab.txt','w') as f: # modified so it don't print a empty line in the last.
		f.write("%s" % vocab[0])
		for word in vocab[1:]:
			f.write("\n%s" % word)
	print("Vocabulary (training set):", len(vocab))
	
	posidict, negadict = train(pos_train, neg_train)
	
	positest = toCounterDictList(pos_test)
	negatest = toCounterDictList(neg_test)

	truePositive = 0
	trueNegative = 0
	falsePositive = 0
	falseNegative = 0

	i = 0
	for posiinstance in positest:
		ppositive = probilityof(pos_train,neg_train,posidict,negadict,'positive',posiinstance,True,True,1)
		pnegative = probilityof(pos_train,neg_train,posidict,negadict,'negative',posiinstance,True,True,1)
		if ppositive > pnegative:
			truePositive += 1
		else:
			falseNegative += 1

	for negainstance in negatest:
		ppositive = probilityof(pos_train,neg_train,posidict,negadict,'positive',negainstance,True,True,1)
		pnegative = probilityof(pos_train,neg_train,posidict,negadict,'negative',negainstance,True,True,1)
		if ppositive < pnegative:
			trueNegative += 1
		else:
			falsePositive += 1
	
	acc = accuracy(truePositive,trueNegative,falsePositive,falseNegative)
	pre = precision(truePositive,trueNegative,falsePositive,falseNegative)
	rec = recall(truePositive,trueNegative,falsePositive,falseNegative)
	f = fscore(truePositive,trueNegative,falsePositive,falseNegative,2)

	print("accuarcy is ",acc)
	print("precision is ",pre)
	print("recall is ",rec)
	print("fscore is ", f)

	return acc

if __name__=="__main__":
	naive_bayes()

