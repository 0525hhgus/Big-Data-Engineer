# 시험환경 세팅 (코드 변경 X)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', 100)

def exam_data_load(df, target, id_name="", null_name=""):
    if id_name == "":
        df = df.reset_index().rename(columns={"index": "id"})
        id_name = 'id'
    else:
        id_name = id_name

    if null_name != "":
        df[df == null_name] = np.nan

    X_train, X_test = train_test_split(df, test_size=0.2, random_state=2021)

    y_train = X_train[[id_name, target]]
    X_train = X_train.drop(columns=[target])

    y_test = X_test[[id_name, target]]
    X_test = X_test.drop(columns=[target])
    return X_train, X_test, y_train, y_test


df = pd.read_csv("diabetes.csv")
x_train, x_test, y_train, y_test = exam_data_load(df, target='Outcome')
y_train = y_train.drop('id', axis=1)

# 데이터 전처리
## 훈련/검증 데이터셋 분리
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.7, test_size=0.3, stratify=y_train, random_state=777)
print(x_train.shape, x_val.shape, y_train.shape, y_val.shape)
x_train = x_train.drop('id', axis=1)
x_val = x_val.drop('id', axis=1)
x_test = x_test.drop('id', axis=1)

## 결측치 처리
print(x_train.isna().sum())
print(x_val.isna().sum())
print(x_test.isna().sum())

## 범주형 변수 인코딩 -> 없음
from sklearn.preprocessing import LabelEncoder
print(x_train.info())
print(x_train.describe())

## 파생변수 만들기

## 표준화 크기
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
rbs = RobustScaler()
scale_col = x_train.columns
print(scale_col)
x_train[scale_col] = rbs.fit_transform(x_train[scale_col])
x_val[scale_col] = rbs.transform(x_val[scale_col])
x_test[scale_col] = rbs.transform(x_test[scale_col])
print(x_train.describe())

## 상관관계
x_corr = x_train.corr()
# print(x_corr)

# 모델링
'''
from sklearn.tree import DecisionTreeClassifier
print(dir(DecisionTreeClassifier))
print(help(DecisionTreeClassifier))
model = DecisionTreeClassifier(max_depth=10, criterion='entropy', random_state=776)
model.fit(x_train, y_train)

from sklearn.ensemble import RandomForestClassifier
print(dir(RandomForestClassifier))
print(help(RandomForestClassifier))
model = RandomForestClassifier(n_estimators=100, max_depth=10, criterion='entropy', random_state=776)
model.fit(x_train, y_train)
'''

from xgboost import XGBClassifier
print(dir(XGBClassifier))
print(help(XGBClassifier))
model = XGBClassifier(n_estimators=100, max_depth=10, criterion='entropy', random_state=776)
model.fit(x_train, y_train)

# 모델 성능 평가
from sklearn.metrics import roc_auc_score
y_val_predict = model.predict_proba(x_val)
y_val_predict = pd.DataFrame(y_val_predict)[1]
print(pd.DataFrame(y_val_predict)[1])
print(y_val_predict)
print(y_val)
print(roc_auc_score(y_val, y_val_predict))

from sklearn.metrics import accuracy_score
y_val_predict = model.predict(x_val)
y_val_predict = pd.DataFrame(y_val_predict)
print(pd.DataFrame(y_val_predict))
print(y_val_predict)
print(y_val)
print(accuracy_score(y_val, y_val_predict))

# 제출
y_pred_result = pd.DataFrame(model.predict(x_test)).rename(columns={0:'OutCome'})

result = pd.concat([y_test['id'], y_pred_result], axis=1)
result.to_csv('./111111.csv', index=False)


