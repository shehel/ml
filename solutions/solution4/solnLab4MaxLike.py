# -*- coding: utf-8 -*-
"""
Created on Thu Feb 02 18:53:50 2017

@author: joe
"""

%matplotlib inline 
# Produce plots in iPython notebook
###################################################################
## call required modules
###################################################################
import pandas as pd
import numpy as np
import math
from numpy.linalg import inv as inverse
import matplotlib.pyplot as plt

###################################################################
## Define functions
###################################################################

# Linear Regression Function for order i polynomial, same as Assignment 1. 
# Produces parameter weights.
def matrixLinReg(dataFrame,xLabel,tLabel,order):
    df = dataFrame.copy() # copy dataframe so don't alter original dataset
    
    #need to add 1s to matrix for intercept calculations - can do in the data frame or in the matrix
    df['1s'] = 1 

    #note that 1s are at the start - will effect which item of matrix w is the intercept
    x = df.as_matrix(columns = ['1s'])
    t = df.as_matrix(columns = [tLabel])

    #for loop to add higher order terms
    for i in range(1,order+1):
        xn = df[xLabel]**i
        x = np.insert(x,i,xn,axis=1)

    #matrix calculations needed
    xT = x.transpose()
    xTx = np.dot(xT,x)
    invxTx = inverse(xTx)
    
    w = np.dot(np.dot(invxTx,xT),t)

    return w

# Extension of Linear regression function to also return Maximum 
# Likelihood Estimate for sample the model variance
def matrixML(dataFrame,xLabel,tLabel,order):
    df = dataFrame.copy() # copy dataframe so don't alter original dataset
    
    #need to add 1s to matrix for intercept calculations - can do in the data frame or in the matrix
    df['1s'] = 1 

    #note that 1s are at the start - will effect which item of matrix w is the intercept
    x = df.as_matrix(columns = ['1s'])
    t = df.as_matrix(columns = [tLabel])

    #for loop to add higher order terms
    for i in range(1,order+1):
        xn = df[xLabel]**i
        x = np.insert(x,i,xn,axis=1)

    #matrix calculations needed
    xT = x.transpose()
    xTx = np.dot(xT,x)
    invxTx = inverse(xTx)
    
    w = np.dot(np.dot(invxTx,xT),t)
    
    tT = t.transpose()
    tTt = np.dot(tT, t)
    tTxw = np.dot(np.dot(tT, x), w)
    N = len(t)
    
    s2 = (tTt - tTxw) / N

    return w, s2

# Function that returns the predictive distribution of at some new x value
def predictML(dataFrame,xLabel,tLabel,order, xPredict):
    df = dataFrame.copy() # copy dataframe so don't alter original dataset
    
    #need to add 1s to matrix for intercept calculations - can do in the data frame or in the matrix
    df['1s'] = 1 

    #note that 1s are at the start - will effect which item of matrix w is the intercept
    x = df.as_matrix(columns = ['1s'])
    t = df.as_matrix(columns = [tLabel])
    #vector to hold the nex x values of each order
    xNew = np.ones(order+1)

    #for loop to add higher order terms
    for i in range(1,order+1):
        xn = df[xLabel]**i
        x = np.insert(x,i,xn,axis=1)
        xNew[i] = xPredict**i 

    #matrix calculations needed
    xT = x.transpose()
    xTx = np.dot(xT,x)
    invxTx = inverse(xTx)
    
    w = np.dot(np.dot(invxTx,xT),t)
    
    #predictive distribution mean at xNew
    tNew = np.dot(xNew, w)
    tNew= tNew.item((0))
    
    tT = t.transpose()
    tTt = np.dot(tT, t)
    tTxw = np.dot(np.dot(tT, x), w)
    N = len(t)
    
    s2 = (tTt - tTxw) / N
    
    xNewT = xNew.transpose()
    
    #predictive distribution variance at xNew
    s2New = s2*np.dot(np.dot(xNewT, invxTx), xNew)
    s2New = s2New.item((0))

    return tNew, s2New
    
###################################################################
## Parameter estimation
###################################################################
male100 = pd.read_csv('male100.csv')

W = matrixLinReg(male100, 'Year', 'Time', 1)

w0 = W.item((0,0))
w1 = W.item((1,0))
    
y = w0 + w1*male100['Year']

print 'Linear Regression of Year vs Winning Time'
male100.plot(x=0,y=1,kind='scatter')
plt.plot(male100['Year'],y,color = 'r')

###################################################################
## Parameter and standard deviation estimation
###################################################################
w, s2 = matrixML(male100, 'Year', 'Time', 1)

w0 = w.item((0,0))
w1 = w.item((1,0))
print w0, w1

std = math.sqrt(s2.item((0,0)))
print std

y = w0 + w1*male100['Year']

print 'Linear Regression of Year vs Winning Time with 1 standard deviation error bars'
male100.plot(x=0,y=1,kind='scatter')
plt.plot(male100['Year'],y,color = 'r')
plt.errorbar(male100['Year'],y,yerr=std, linestyle="None")

###################################################################
## Predictive distribution estimation
###################################################################
years = np.array(range(1896, 2052, 4))
times = np.zeros(len(years))
pred_error = np.zeros(len(years))

j=0
for i in years:
    times[j], pred_error[j] =   predictML(male100, 'Year', 'Time', 1, i)
    j = j + 1
    
print 'Linear Regression of Year vs Winning Time with 1 standard deviation of predictive distribution error bars'
male100.plot(x=0,y=1,kind='scatter')
plt.plot(years,times,color = 'r')
plt.errorbar(years,times,yerr=2*np.sqrt(pred_error),color = 'b', linestyle="None")

