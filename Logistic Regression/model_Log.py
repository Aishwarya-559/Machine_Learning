
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from patsy.highlevel import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,accuracy_score,roc_auc_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from pandas_profiling import ProfileReport
import pickle
import multiprocessing
from multiprocessing import Process, freeze_support


dta = sm.datasets.fair.load_pandas().data
print(dta)
ProfileReport(dta)
dta['Affair'] = (dta.affairs > 0).astype(int)
print(dta.head())

x = dta.drop(columns=['Affair'])
y = dta['Affair']
scaler = StandardScaler()
scaled_data = scaler.fit_transform(x)
print(scaled_data)
print(pd.DataFrame(scaled_data))
print(dta.columns)


df = pd.DataFrame(scaled_data, columns=['rate_marriage', 'age', 'yrs_married', 'children', 'religious', 'educ','occupation', 'occupation_husb', 'affairs'])

print(df.head())

from statsmodels.stats.outliers_influence import variance_inflation_factor

df1 = pd.DataFrame()
df1['VIF']=[variance_inflation_factor(scaled_data,i) for i in range(scaled_data.shape[1])]
df1['feature'] = x.columns
print(df1)

#As the yrs_married has vif of 7.1 we'll remove that.
df3 = df.drop(columns=['yrs_married'], inplace=True)
x = df
y = dta.Affair
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25, random_state=43)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
log = LogisticRegression()
log.fit(x_train,y_train)
print(log.coef_)
print(log.intercept_)

pred = log.predict(x_test)
print(pred)
prob = log.predict_proba(x_test)
print(prob)
cm = confusion_matrix(y_test,pred)
print(cm)
print(f'Accuracy score is: {accuracy_score(y_test,pred)}')
print(f'classification_report: ', classification_report(y_test,pred))
print(f'The roc_auc_score is: ', roc_auc_score(y_test,pred))
score = cross_val_score(LogisticRegression(), x,y, scoring='accuracy', cv=10)
score, score.mean()

#
# pickle.dump(log,open('Log.pkl', 'wb'))
#
# model = pickle.load(open('Log.pkl','rb'))

import pickle

pickle.dump(log,open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))