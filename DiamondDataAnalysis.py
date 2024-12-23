import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

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
df.shape

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

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)

train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
# Plot the first graph
axes[0].scatter(y_train, train_pred, alpha=0.5)
axes[0].plot(np.linspace(0, np.max(y_train), 100), np.linspace(0, np.max(y_train), 100), '--', color='red', label='Prediction Line')
axes[0].set_title('Train Data')
axes[0].set_xlabel('Actual Values (y_test)')
axes[0].set_ylabel('Predicted Values')
axes[0].legend()
axes[0].grid(True)
# Plot the second graph
axes[1].scatter(y_test, test_pred, alpha=0.5)
axes[1].plot(np.linspace(0, np.max(y_test), 100), np.linspace(0, np.max(y_test), 100), '--', color='red', label='Prediction Line')
axes[1].set_title('Test Data')
axes[1].set_xlabel('Actual Values (y_test)')
axes[1].set_ylabel('Predicted Values')
axes[1].legend()
axes[1].grid(True)
# Adjust layout
plt.tight_layout()
plt.show()

coefficients = model.coef_
column_names = X_train.columns
coefficients_df = pd.DataFrame({'Column Name': column_names, 'Coefficient': coefficients})
coefficients_df



from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("::---------Train Data---------::")
mae = mean_absolute_error(y_train,train_pred)
mse = mean_squared_error(y_train,train_pred)
r2_train = r2_score(y_train,train_pred)
print('Mean Absolute Error:',mae)
print('Mean Squared Error:',mse)
print('R2 Score:',r2_train*100,'%')
print('*'*50)
print("::---------Test Data---------::")
mae = mean_absolute_error(y_test,test_pred)
mse = mean_absolute_error(y_test,test_pred)
r2_test = r2_score(y_test,test_pred)
print('Mean Absolute Error:',mae)
print('Mean Squared Error:',mse)
print('R2 Score:',r2_test*100,'%')
print('*'*50)