import numpy as np
import pandas as pd
import os

#C:\Users\singh\.cache\kagglehub\datasets\shivam2503\diamonds\versions\1\diamonds.csv

train = pd.read_csv(r'C:\Users\singh\.cache\kagglehub\datasets\shivam2503\diamonds\versions\1\diamonds.csv')
df=train.copy()
df.head()
df.tail()