import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import seaborn

#C:\Users\singh\.cache\kagglehub\datasets\shivam2503\diamonds\versions\1\diamonds.csv

df = pd.read_csv(r'C:\Users\singh\.cache\kagglehub\datasets\shivam2503\diamonds\versions\1\diamonds.csv')

df.head()

df.describe()
