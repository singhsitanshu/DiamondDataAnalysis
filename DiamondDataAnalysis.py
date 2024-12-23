import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import seaborn

#C:\Users\singh\.cache\kagglehub\datasets\shivam2503\diamonds\versions\1\diamonds.csv

df = pd.read_csv(r'C:\Users\singh\.cache\kagglehub\datasets\shivam2503\diamonds\versions\1\diamonds.csv')

df.head()

df.describe()

df.drop(columns='Unnamed: 0', axis=1, inplace=True)
df.head()

df.shape

df['cut'].unique()
df['color'].unique()
df['clarity'].unique()

s_25 = df['price'].quantile(0.25)
s_75 = df['price'].quantile(0.75)
IQR = s_75-s_25

s_lower = max(s_25 - IQR*1.5, df['price'].min())
s_upper = min(s_75 + IQR*1.5, df['price'].max())
df = df[(df['price']>=s_lower) & (df['price']<=s_upper)]
df.shape()

df.isna().sum()*100/df.shape[0]

count=len(df[df.duplicated()])
if count>0:
    print('Duplicate Data Count:',count)
    df.drop_duplicates(inplace=True)
    print('Dropped!')