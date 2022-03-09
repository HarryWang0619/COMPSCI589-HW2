from utils import *
from pprint import pprint
import time

def naive_bayes(ptrain:float=0.1,ntrain:float=0.1,ptest:float=0.1,ntest:test=0.1,laplacesmooth:bool=true, logbool:bool=true, smoothconst:float=1):
	
	t0 = time.time()

	percentage_positive_instances_train = ptrain
	percentage_negative_instances_train = ntrain

	percentage_positive_instances_test  = ptest
	percentage_negative_instances_test  = ntest

	print("Loading Data, at time ", (0))

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
	
	print("Learning Training Data, at time ", (time.time()-t0))

	posidict, negadict = train(pos_train, neg_train)
	
	positest = toCounterDictList(pos_test)
	negatest = toCounterDictList(neg_test)

	truePositive = 0
	trueNegative = 0
	falsePositive = 0
	falseNegative = 0

	# i = 0
	print("Now testing... at time ", (time.time()-t0))
	print("Testing Positive")
	for posiinstance in positest:
		ppositive = probilityof(pos_train,neg_train,posidict,negadict,'positive',posiinstance,logbool,laplacesmooth,smoothconst)
		pnegative = probilityof(pos_train,neg_train,posidict,negadict,'negative',posiinstance,logbool,laplacesmooth,smoothconst)
		if ppositive > pnegative:
			truePositive += 1
		else:
			falseNegative += 1

	print("Testing Negative, at time ", (time.time()-t0))
	for negainstance in negatest:
		ppositive = probilityof(pos_train,neg_train,posidict,negadict,'positive',negainstance,logbool,laplacesmooth,smoothconst)
		pnegative = probilityof(pos_train,neg_train,posidict,negadict,'negative',negainstance,logbool,laplacesmooth,smoothconst)
		if ppositive < pnegative:
			trueNegative += 1
		else:
			falsePositive += 1
	
	acc = accuracy(truePositive,trueNegative,falsePositive,falseNegative)
	pre = precision(truePositive,trueNegative,falsePositive,falseNegative)
	rec = recall(truePositive,trueNegative,falsePositive,falseNegative)
	f = fscore(truePositive,trueNegative,falsePositive,falseNegative,1)

	print("accuarcy is ",acc)
	print("precision is ",pre)
	print("recall is ",rec)
	print("fscore is ", f)
	print("total time cose is ", (time.time()-t0), "sec")
	return acc

if __name__=="__main__":
	naive_bayes(0.3,0.3,0.5,0.5)

