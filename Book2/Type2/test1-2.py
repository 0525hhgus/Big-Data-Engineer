import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 100)

df = pd.read_csv('Bank_Personal_Loan_Modelling.csv')
print(df.shape)
print(df.head())


df = df.dropna().drop(['ID', 'ZIP Code'], axis=1)
df_x = df.drop(['Personal Loan'], axis=1)
df_y = df['Personal Loan']


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, stratify=df_y, train_size=0.7, test_size=0.3, random_state=1234)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

import sklearn.preprocessing as preprecessing

preprecessor = preprecessing.Normalizer()
x_train = preprecessor.fit_transform(x_train)
x_test = preprecessor.transform(x_test)

from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

train_acc = []
test_acc = []

kneighbors_settings = range(1, 25)

for n_neighbors in kneighbors_settings:
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(x_train, y_train)

    train_acc.append(clf.score(x_train, y_train))
    test_acc.append(clf.score(x_test, y_test))

plt.plot(kneighbors_settings, train_acc, label="Train Accuracy")
plt.plot(kneighbors_settings, test_acc, label="Test Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
plt.show()

print(test_acc)



