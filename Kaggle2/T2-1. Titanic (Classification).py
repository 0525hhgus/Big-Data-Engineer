import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 100)

df = pd.read_csv('titanic.csv')

print(df.head())
print(df.shape)

# 데이터전처리
df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
print(df.isna().sum())
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
# df['Cabin'] = df['Cabin'].fillna(df['Cabin'].mode()[0])
print(df.isna().sum())

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
le = LabelEncoder()
df['SibSp'] = le.fit_transform(df['SibSp'])
# le2 = LabelEncoder()
# df['Cabin'] = le2.fit_transform(df['Cabin'])

df_onehot = pd.get_dummies(df, columns=['Pclass', 'Sex', 'Embarked'])
print(df_onehot.head())

from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
df_onehot[['Age', 'Fare']] = mms.fit_transform(df_onehot[['Age', 'Fare']])

print(df_onehot.head())



df_x = df_onehot.drop(['Survived'], axis=1)
df_y = df_onehot['Survived']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.3, train_size=0.7, stratify=df_y, random_state=333)

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=200, max_depth=7, random_state=333)
clf.fit(x_train, y_train)

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

pred = clf.predict(x_test)
print(clf.score(x_test, y_test))

print(accuracy_score(pred, y_test))
print(confusion_matrix(pred, y_test))





