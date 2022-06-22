import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 100)

df = pd.read_csv('Parkinsson disease.csv')
print(df.head())

df = df.drop('name', axis=1, inplace=False)

import sklearn.preprocessing as preprocessing
df_norm = preprocessing.minmax_scale(df)
df_processed = pd.DataFrame(df_norm, columns=df.columns)

import statsmodels.api as sm
# 상수항 추가
df_const = sm.add_constant(df_processed, has_constant='add')

df_x = df_const.drop('status', axis=1)
df_y = df_const['status'].astype('category')

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, stratify=df_y, test_size=0.1, random_state=2022)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)


model = sm.Logit(y_train, x_train)
results = model.fit(method='bfgs', maxiter=1000)
print(results.summary())


def cut_off(y, threshold):
    Y = y.copy()
    Y[Y>threshold] = 1
    Y[Y<=threshold] = 0
    return Y.astype(int)

test_y_pred_prob = results.predict(x_test)
test_y_pred = cut_off(test_y_pred_prob, 0.8)

from sklearn.metrics import f1_score
print(f1_score(y_test, test_y_pred))




