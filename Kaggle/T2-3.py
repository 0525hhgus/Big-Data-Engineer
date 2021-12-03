import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', 50)

df = pd.read_csv('adult.csv')

print(df.info())
df['income'] = df['income'].replace('<=50K', 0).replace('>50K', 1)
# print(df['income'].unique())

# print(df.isnull().sum())
print(df['workclass'].unique())
df['workclass'] = df['workclass'].replace('?', np.NaN)
# print(df['workclass'].unique())
# print(df['workclass'].isnull().sum())
print(df['occupation'].unique())
df['occupation'] = df['occupation'].replace('?', np.NaN)
print(df['native.country'].unique())
df['native.country'] = df['native.country'].replace('?', np.NaN)

col = df.columns
for c in col[:-1]:
    print(df.groupby([c])['income'].sum())
df = df.drop(['fnlwgt'], axis=1)

# 결측치
df['workclass'] = df['workclass'].fillna('Private')
df['occupation'] = df['occupation'].fillna('Exec-managerial')
df['native.country'] = df['native.country'].fillna('United-States')

from sklearn.preprocessing import LabelEncoder, StandardScaler
scaler = StandardScaler()
df[['capital.gain', 'capital.loss']] = scaler.fit_transform(df[['capital.gain', 'capital.loss']])
'''
encoder = LabelEncoder()
col_obj = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']
for c in col_obj:
    df[c] = encoder.fit_transform(df[c])
'''
col_obj = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex']
for c in col_obj:
    df = pd.concat([df, pd.get_dummies(df[c])], axis=1)
    df = df.drop(c, axis=1)

encoder = LabelEncoder()
df['native.country'] = encoder.fit_transform(df['native.country'])

print(df.columns)
df_x = df.drop(['income'], axis=1)

x_train, x_test, y_train, y_test = train_test_split(df_x, df['income'], test_size=0.2, random_state=10)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

from xgboost import XGBClassifier
xgb_model = XGBClassifier(n_estimates = 200, max_depth = 10, random_state=10)
xgb_model.fit(x_train, y_train)

from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
xgb_pred = xgb_model.predict(x_test)
# print(y_test.value_counts())
# print(pd.DataFrame(xgb_pred).value_counts())
# print(y_test)
# print(xgb_pred)
print(roc_auc_score(y_test, xgb_pred))
print(classification_report(y_test, xgb_pred))

from sklearn.ensemble import RandomForestClassifier
rmf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=10)
rmf_model.fit(x_train, y_train)

rmf_pred = rmf_model.predict(x_test)
print(roc_auc_score(y_test, rmf_pred))
print(classification_report(y_test, rmf_pred))

result = pd.concat([pd.DataFrame(xgb_pred), pd.DataFrame(rmf_pred)], axis=1)
result.to_csv('000000.csv', index=False)

