import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv('https://raw.githubusercontent.com/BigDataGal/Python-for-Data-Science/master/titanic-train.csv',
                 usecols=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Survived'])

print(df.head())
print(df.isnull().sum())
df['Age'] = df.Age.fillna(df['Age'].mean())

x = df.drop(columns=['Survived'])
y = df['Survived']

x['Sex'] = pd.get_dummies(x['Sex']).values

print(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=100)

model = DecisionTreeClassifier()

model.fit(x_train, y_train)
predict = model.predict(x_test)
print("Accuracy score is: ", accuracy_score(y_test, predict))
prediction = pd.DataFrame({'Actual': y_test, 'Predicted': predict})
print(prediction)

print('Training Score: ', model.score(x_train, y_train))
print('Testing Score: ', model.score(x_test, y_test))

path = model.cost_complexity_pruning_path(x_train, y_train)
ccp_alpha = path.ccp_alphas

print(ccp_alpha)

dt_model = []
for ccp in ccp_alpha:
    dt_m = DecisionTreeClassifier(ccp_alpha=ccp)
    dt_m.fit(x_train, y_train)
    dt_model.append(dt_m)

print(dt_model)
train_score = [i.score(x_train, y_train) for i in dt_model]
test_score = [i.score(x_test, y_test) for i in dt_model]

fix, ax = plt.subplots()
ax.set_xlabel('alpha')
ax.set_ylabel('accuracy')
ax.plot(ccp_alpha, train_score, marker='o', label='train')
ax.plot(ccp_alpha, test_score, marker='o', label='test')
ax.legend()
plt.show()

dt_model_ccp = DecisionTreeClassifier(ccp_alpha=0.037)
dt_model_ccp.fit(x_train, y_train)

print(dt_model_ccp.score(x_train, y_train))
print(dt_model_ccp.score(x_test, y_test))

from sklearn import tree

plt.figure(figsize=(20, 20))
tree.plot_tree(dt_model_ccp, feature_names=x.columns, filled=True, class_names=[str(i) for i in set(y)])
plt.show()

import pickle

pickle.dump(dt_model_ccp, open('Decision_tree.pickle', 'wb'))
model = pickle.load(open('Decision_tree.pickle', 'rb'))
