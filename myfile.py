%run myfile.py

import pandas as pd
import numpy as np

male100 = pd.read_csv('male100.csv', header = 0)
print male100

male100.to_csv('demo.csv')

copymale100 = male100.copy()

#mean = copymale100[’Time’].mean()
#std = copymale100[’Time’].std()
#To get some basic statistics, we can use the describe() method:
#print copymale100[’Time’].describe(), "\n"
#print mean, std