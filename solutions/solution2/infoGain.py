from __future__ import division
import pandas as pd
import math


def entropy(dataFrame, pos_val, neg_val,tLabel):
    npos = (dataFrame[tLabel] == pos_val).sum()
    nneg = (dataFrame[tLabel] == neg_val).sum()
    n = npos + nneg
    
    if npos == 0 or nneg == 0:
        entropy = 0
    else:
        entropy = -((npos/n)*math.log((npos/n),2) + (nneg/n)*math.log((nneg/n),2))

    return entropy


def infoGain(dataFrame, attr, pos_val, neg_val, tLabel):
    sEntropy = entropy(dataFrame,pos_val, neg_val, tLabel)
    sSize = len(dataFrame)
    sumEnt = 0

    for i in dataFrame[attr].unique():
        siSize = (dataFrame[attr] == i).sum()
        #temp = (siSize/sSize)*entropy(dataFrame.loc[dataFrame[attr]==i],pos_val,neg_val,tLabel)
        
        sumEnt += (siSize/sSize)*entropy(dataFrame.loc[dataFrame[attr]==i],pos_val,neg_val,tLabel)
    
    infogain = sEntropy - sumEnt
    return infogain


dataFrame = pd.read_csv('playTennis.csv',header = 0)

print entropy(dataFrame,'yes','no','play')
print infoGain(dataFrame,'outlook','yes','no','play')


# calc info gain for all attributes (but not the class label)
#for i in list(dataFrame)[:-1]: # the class label may not always be the last element- but you should know the attr name
for i in list(dataFrame.drop('play',axis=1)):
    print i
    print infoGain(dataFrame,i,'yes','no','play')

