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

df.head()

X = df.drop(columns='price',axis=1)
y = df['price']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y ,test_size=0.2, random_state=42)

from category_encoders.target_encoder import TargetEncoder

encoder = TargetEncoder()
columns = ['cut','color','clarity']
for column in columns:
    X_train[column] = encoder.fit_transform(X = X_train[column], y = y_train)
    X_test[column] = encoder.fit_transform(X = X_test[column], y = y_test)

X_train.head()

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
# Train data split
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
# Test data split
X_test = pd.DataFrame(scaler.fit_transform(X_test), columns=X_test.columns)