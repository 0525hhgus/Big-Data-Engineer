import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

pd.set_option('display.max_columns', 100)

df_x = pd.read_csv('titanic_x_train.csv', encoding='cp949')
df_y = pd.read_csv('titanic_y_train.csv', encoding='cp949')
df_test = pd.read_csv('titanic_x_test.csv', encoding='cp949')

x_train, x_test, y_train, y_test = train_test_split(df_x, df_y,  test_size=0.3, random_state=10)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
# print(y_train)
y_train.columns = ['PassengerId', 'Survived']
y_test.columns = ['PassengerId', 'Survived']

x_train_id = x_train['PassengerId']
x_test_id = x_test['PassengerId']
y_train_id = y_train['PassengerId']
y_test_id = y_test['PassengerId']
df_test_id = df_test['PassengerId']

x_train = x_train.drop(['PassengerId'], axis=1)
x_test = x_test.drop(['PassengerId'], axis=1)
y_train = y_train.drop(['PassengerId'], axis=1)
y_test = y_test.drop(['PassengerId'], axis=1)
df_test = df_test.drop(['PassengerId'], axis=1)

df_y.columns = ['PassengerId', 'Survived']
df_y = df_y.drop(['PassengerId'], axis=1)
df_x = df_x.drop(['PassengerId'], axis=1)
data = pd.concat([df_x, df_y], axis=1)

# 데이터 전처리
print(data.info())
x_train = x_train.drop(['승객이름'], axis=1)
x_test = x_test.drop(['승객이름'], axis=1)
df_test = df_test.drop(['승객이름'], axis=1)
data = data.drop(['승객이름'], axis=1)

# 결측치: 나이, 객실번호, 선착장
print(data.isnull().sum())
# print(data['객실번호'].unique())
# print(data['선착장'].unique())
print(data['티켓번호'].unique().size)
x_train = x_train.drop(['티켓번호'], axis=1)
x_test = x_test.drop(['티켓번호'], axis=1)
df_test = df_test.drop(['티켓번호'], axis=1)
data = data.drop(['티켓번호'], axis=1)

## 나이
print(data['나이'].isnull().sum())
print(data[['나이', 'Survived']].corr()) # -0.077221
x_train = x_train.drop(['나이'], axis=1)
x_test = x_test.drop(['나이'], axis=1)
df_test = df_test.drop(['나이'], axis=1)
data = data.drop(['나이'], axis=1)

## 객실번호
print(data['객실번호'].isnull().sum())
print(data['객실번호'].unique().size)
x_train = x_train.drop(['객실번호'], axis=1)
x_test = x_test.drop(['객실번호'], axis=1)
df_test = df_test.drop(['객실번호'], axis=1)
data = data.drop(['객실번호'], axis=1)

## 선착장
print(data['선착장'].isnull().sum())
print(data.groupby(['선착장'])['선착장'].count())
x_train['선착장'] = x_train['선착장'].fillna('S')
x_test['선착장'] = x_test['선착장'].fillna('S')
df_test['선착장'] = df_test['선착장'].fillna('S')
data['선착장'] = data['선착장'].fillna('S')

print(data.isnull().sum())

# 범주형 변수 인코딩
print(data.info())
'''
encoder = LabelEncoder()
x_train['성별'] = encoder.fit_transform(x_train['성별'])
x_test['성별'] = encoder.fit_transform(x_test['성별'])
df_test['성별'] = encoder.fit_transform(df_test['성별'])
data['성별'] = encoder.fit_transform(data['성별'])

'''
x_train['성별'] = x_train['성별'].replace('male',0).replace('female',1)
x_test['성별'] = x_test['성별'].replace('male',0).replace('female',1)
df_test['성별'] = df_test['성별'].replace('male',0).replace('female',1)
data['성별'] = data['성별'].replace('male',0).replace('female',1)

cqs_dummy = pd.get_dummies(x_train['선착장'], drop_first=True).rename(columns={'Q': '선착장Q', 'S': '선착장S'})
x_train = pd.concat([x_train, cqs_dummy], axis=1)
cqs_dummy = pd.get_dummies(x_test['선착장'], drop_first=True).rename(columns={'Q': '선착장Q', 'S': '선착장S'})
x_test = pd.concat([x_test, cqs_dummy], axis=1)
cqs_dummy = pd.get_dummies(df_test['선착장'], drop_first=True).rename(columns={'Q': '선착장Q', 'S': '선착장S'})
df_test = pd.concat([df_test, cqs_dummy], axis=1)
cqs_dummy = pd.get_dummies(data['선착장'], drop_first=True).rename(columns={'Q': '선착장Q', 'S': '선착장S'})
data = pd.concat([data, cqs_dummy], axis=1)

x_train = x_train.drop(['선착장'], axis=1)
x_test = x_test.drop(['선착장'], axis=1)
df_test = df_test.drop(['선착장'], axis=1)
data = data.drop(['선착장'], axis=1)

print(data.info())

## 파생변수
x_train['가족수'] = x_train['형제자매배우자수'] + x_train['부모자식수']
x_test['가족수'] = x_test['형제자매배우자수'] + x_test['부모자식수']
df_test['가족수'] = df_test['형제자매배우자수'] + df_test['부모자식수']
data['가족수'] = data['형제자매배우자수'] + x_train['부모자식수']
x_train = x_train.drop(['형제자매배우자수', '부모자식수'], axis=1)
x_test = x_test.drop(['형제자매배우자수', '부모자식수'], axis=1)
df_test = df_test.drop(['형제자매배우자수', '부모자식수'], axis=1)
data = data.drop(['형제자매배우자수', '부모자식수'], axis=1)
print(data.info())

# 학습
model = XGBClassifier(n_estimators=100, max_depth=5, eval_metric = 'error', random_state=10)
model.fit(x_train, y_train)

'''
y_pred = model.predict(x_test)
print(roc_auc_score(y_test, y_pred))

y_pred_result = pd.DataFrame(y_pred).rename(columns={0:'Survived'})

y_test_id = y_test_id.reset_index()
result = pd.concat([y_test_id, y_pred_result], axis=1)
result.to_csv('./111111.csv', index=False)

df_final = pd.read_csv('111111.csv')
df_final = df_final['Survived']
print(roc_auc_score(y_test, df_final))
'''

y_pred_result = pd.DataFrame(model.predict(df_test)).rename(columns={0:'Survived'})

result = pd.concat([df_test_id, y_pred_result], axis=1)
result.to_csv('./111111.csv', index=False)
