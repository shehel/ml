
# coding: utf-8

# In[21]:

import math
import pandas as pd
import numpy as np
import sklearn.linear_model as sk


# In[22]:

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


# In[23]:

dfProbeA = pd.read_csv('../probeA.csv', header = 0)
dfScProbeA = scaleData(dfProbeA, 2)

dfProbeB = pd.read_csv('../probeB.csv', header = 0)
dfScProbeB = scaleData(dfProbeB, 2)

dfClassA = pd.read_csv('../classA.csv', header = 0)


# In[24]:

dfTarget = dfProbeA[['TNA']]
dfScProbeA = dfScProbeA.drop('TNA',axis=1)


# In[25]:

ridge = sk.RidgeCV(alphas=np.arange(-0.1,0.01,0.0005))
ridge.fit(dfScProbeA, dfTarget)
predicted = ridge.predict(dfScProbeB)
np.savetxt("tnaB.csv", predicted, delimiter=",")


# In[ ]:



