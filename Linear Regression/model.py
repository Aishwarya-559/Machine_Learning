import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import load_boston
import pickle

d = sklearn.datasets.load_boston()

df = pd.DataFrame(d.data, columns= d.feature_names)
print(df)
df['MEDV'] = d.target
print(df.isnull().sum())
print(df.describe())

plt.subplots(figsize = (20,10))
sns.distplot(df['MEDV'],hist=True,kde=True)

sns.boxplot(data=df['MEDV'])

corr = df.corr()
plt.subplots(figsize = (15,10))
sns.heatmap(corr,annot=True, cmap= 'rainbow')

plt.figure(figsize=(20,5))


features = ['LSTAT','RM']
target = df['MEDV']

for i,col in enumerate(features):
    plt.subplot(1, len(features),i+1)
    x = df[col]
    y = target
    plt.scatter(x,y)
    plt.xlabel(col)
    plt.ylabel(col)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn import metrics

x = df.iloc[:,:13]
y = df.iloc[:,13]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

lr = LinearRegression()
lr.fit(x_train,y_train)
print(lr.intercept_)

slope = pd.DataFrame(lr.coef_, x.columns,columns=['Coefficient'])
print(slope)

y_pred = lr.predict(x_test)
print(y_pred)

plt.scatter(y_test,y_pred)

results = pd.DataFrame({'Actual':y_test, 'Predicted': y_pred})
results.head()

print(f'MAE:{metrics.mean_absolute_error(y_test, y_pred)}')
print(f'MSE:{metrics.mean_squared_error(y_test, y_pred)}')
print(f'RMSE:{np.sqrt(metrics.mean_squared_error(y_test, y_pred))}')

import pickle

pickle.dump(lr,open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))