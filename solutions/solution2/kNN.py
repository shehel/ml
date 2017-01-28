from __future__ import division
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn import cross_validation as cv

def calcAcc(pred,t):
    TPTN = 0
    total = 0

    for index,val in enumerate(pred):
        if val == t.iloc[index]:
            TPTN += 1
        total += 1
        
    return TPTN, total


#predict time given model and year for instances in the test set
def predict(model,testDF,xLabel,tLabel):
    testData = testDF.copy()[xLabel]
    pred = model.predict(testData)

    return calcAcc(pred,testDF[tLabel])


def crossValidation(dataFrame,nbrs):
    #kf contains indices of instances in each fold - use KFold to split
    kf = cv.KFold(len(dataFrame),10)
    TPTN = 0
    total = 0

    #KFold returns a list containing the training instances and testing instances in each list item
    #will have 10 items in the list if using 10-fold CV
    for train, test in kf:
        trainingDF = dataFrame.iloc[train].copy()
        testingDF = dataFrame.iloc[test].copy()
        
        n = kNN(n_neighbors=nbrs)
        n.fit(trainingDF[range(0,8)],trainingDF[8])

        t1, t2 = predict(n,testingDF,range(0,8),8)

        TPTN += t1
        total += t2
    
    return TPTN/total


dataFrame = pd.read_csv('diabetes.data', header = None)

means = dataFrame.mean()
stds = dataFrame.std()

#standarise attributes - DO NOT INCLUDE THE CLASS LABEL
dataFrame[range(0,8)] = (dataFrame[range(0,8)] - means[range(0,8)])/stds[range(0,8)]


k1 = crossValidation(dataFrame,1)
print k1

k3 = crossValidation(dataFrame,3)
print k3

k5 = crossValidation(dataFrame,5)
print k5
