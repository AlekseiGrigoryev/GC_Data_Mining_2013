__author__ = 'Alexey Grigoryev'

from sklearn.ensemble import RandomForestClassifier
from numpy import savetxt, loadtxt
from sklearn.externals import joblib
from sklearn import cross_validation
import numpy as np
import logloss

def recognize (train_file,test_file):
    print ("1.Parsing sets")
    datalist = loadtxt(open(train_file, 'r'), dtype='f8', delimiter=',', skiprows=1)
    joblib.dump(datalist, 'training_set.pkl')
    datalist = joblib.load('training_set.pkl')
    label = [x[0] for x in datalist]
    train = [x[1:] for x in datalist]
    test = loadtxt(open(test_file, 'r'), dtype='f8', delimiter=',', skiprows=1)
    joblib.dump(test, 'test_set.pkl')
    test = joblib.load('test_set.pkl')

    print ("2.Create and train RF")
    temp = RandomForestClassifier(n_estimators=100, n_jobs=4)
    cv = cross_validation.KFold(len(train), n_folds=5, indices=True)

    scores = []
    #results = []
    #for traincv, testcv in cv:
       # trainfit = train[testcv[0]:testcv[-1]+1]
       # print (len(trainfit))
       # traintest = train [:]
       # print (len(traintest))
       # del traintest[testcv[0]:testcv[-1]+1]
       # labelfit = label[testcv[0]:testcv[-1]+1]
       # labeltest = label[:]
       # del labeltest[testcv[0]:testcv[-1]+1]
       # probas = temp.fit(trainfit, labelfit).predict_proba(traintest)
       # results.append( logloss.llfun(labeltest, [x[1] for x in probas]) )
    for train_indices, test_indices in cv:
        print ('Train: %s | test: %s' % (len(train_indices),len(test_indices)))
        trainfit = train[train_indices[0]:train_indices[-1]+1]
        traintest = train [test_indices[0]:test_indices[-1]+1]
        labelfit = label[train_indices[0]:train_indices[-1]+1]
        labeltest = label[test_indices[0]:test_indices[-1]+1]
        scores.append(temp.fit(trainfit, labelfit).score(traintest, labeltest))

    print ("Accuracy: " + str(np.array(scores).mean()))
    print ("Scores" + scores)
    #print ("Results: " + str( np.array(results).mean() ))

if __name__ == "__main__":
    recognize ("train.csv","test.csv")