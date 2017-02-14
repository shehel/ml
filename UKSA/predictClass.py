
# coding: utf-8

# In[11]:

import math
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


# In[12]:

def scaleData(dataFrame,flag):
    df = dataFrame.copy()

    for var in df:
        mean = df[var].mean()
        std = df[var].std()
        l1 = (df[var].abs()).sum()

        if(flag == 1):
            df[var] = (df[var]-mean)/std
        else:
            df[var] = df[var]/l1

    return df


# In[16]:

dfProbeA = pd.read_csv('../probeA.csv', header = 0)
dfProbeB = pd.read_csv('../probeB.csv', header = 0)
dfClassA = pd.read_csv('../classA.csv', header = 0)
dfScProbeA = scaleData(dfProbeA,1)
dfScProbeB = scaleData(dfProbeB,1)

dfScProbeA = dfScProbeA.drop('TNA', 1)
dfScProbeA = dfScProbeA.drop('c1', 1)
dfScProbeA = dfScProbeA.drop('n2', 1)
dfScProbeA = dfScProbeA.drop('m3', 1)
dfScProbeA = dfScProbeA.drop('p1', 1)
dfScProbeA = dfScProbeA.drop('m1', 1)

dfScProbeB = dfScProbeB.drop('c1', 1)
dfScProbeB = dfScProbeB.drop('n2', 1)
dfScProbeB = dfScProbeB.drop('m3', 1)
dfScProbeB = dfScProbeB.drop('p1', 1)
dfScProbeB = dfScProbeB.drop('m1', 1)


# In[17]:

knn = KNeighborsClassifier(n_neighbors=22, weights='distance')
knn.fit(dfScProbeA, dfClassA.values.ravel())

predicted = knn.predict(dfScProbeB)

np.savetxt("classB.csv", predicted, delimiter=",")


# In[ ]:



