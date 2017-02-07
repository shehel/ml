import pandas as pd
import numpy as np
import sklearn.linear_model as lm

def scaleData(dataFrame,notScaled):
    df = dataFrame.copy()

    for var in df.drop(notScaled,axis=1):
        mean = df[var].mean()
        std = df[var].std()

        df[var] = (df[var]-mean)/std

    return df

#alternative way to normalise data
def scaleData2(dataFrame,notScaled):
    df = dataFrame.copy()
    header = list(df.columns.values)
    for var in notScaled:
        header.remove(var)

    df[header] = df[header].apply(lambda x: (x - x.mean()) / x.std())
    return df


def getRsquared(model,training,testing,tLabel):
    model.fit(training.drop(tLabel,axis=1),training[tLabel])
    r2 = model.score(testing.drop(tLabel,axis=1),testing[tLabel])
    print r2



dataFrame = pd.read_table('prostate.data',header=0, index_col=0)
dataFrame = scaleData2(dataFrame,['lpsa','svi','train'])

trainData = dataFrame[dataFrame['train']=='T'].copy()
trainData = trainData.drop('train',axis=1)

testData = dataFrame[dataFrame['train']=='F'].copy()
testData = testData.drop('train',axis=1)


#OLS model - using default parameters
lrModel = lm.LinearRegression()
getRsquared(lrModel,trainData,testData,'lpsa')

#Ridge model - use CV varient to set best regularisation strength
ridgeModel = lm.RidgeCV(alphas=(0.1,1.0,10.0),cv = 10)
getRsquared(ridgeModel,trainData,testData,'lpsa')

#Lasso model
lassoModel = lm.LassoCV()
getRsquared(lassoModel,trainData,testData,'lpsa')
