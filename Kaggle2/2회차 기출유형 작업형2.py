import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 100)

df = pd.read_csv('e-commerce.csv') # 종속 변수: Reached.on.Time_Y.N
print(df.head())
print(df.shape)

# 데이터전처리
## 데이터셋 분리
from sklearn.model_selection import train_test_split
train = df.drop('Reached.on.Time_Y.N', axis=1)
test = df['Reached.on.Time_Y.N']
x_train, x_test, y_train, y_test = train_test_split(train, test, test_size=0.3, stratify=test, random_state=555)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.3, stratify=y_train, random_state=555)
print(x_train.shape, x_val.shape, x_test.shape)
x_train = x_train.drop('ID', axis=1)
x_val = x_val.drop('ID', axis=1)

## 데이터셋 유형
print(x_train.info())
print(x_train.describe())
print(x_train[['Warehouse_block', 'Mode_of_Shipment', 'Product_importance', 'Gender']].head()) # 범주형 변수
col_cat = ['Warehouse_block', 'Mode_of_Shipment', 'Product_importance', 'Gender']
for i in col_cat:
    print(x_train[i].value_counts())


## 결측치 처리
# print(df.isna().sum()) # 결측치 없음

## 스케일링
from sklearn.preprocessing import StandardScaler
sds = StandardScaler()
col_scale = ['Prior_purchases', 'Discount_offered', 'Weight_in_gms']
x_train[col_scale] = sds.fit_transform(x_train[col_scale])
x_val[col_scale] = sds.transform(x_val[col_scale])
x_test[col_scale] = sds.transform(x_test[col_scale])
print(x_train.describe())

## 라벨 인코딩, 원핫 인코딩
# col_cat = ['Warehouse_block' 3, 'Mode_of_Shipment' 3, 'Product_importance' 3, 'Gender' 2]
from sklearn.preprocessing import LabelEncoder

for i in col_cat:
    le = LabelEncoder()
    x_train[i] = le.fit_transform(x_train[i])
    x_val[i] = le.transform(x_val[i])
    x_test[i] = le.transform(x_test[i])
print(x_train[['Warehouse_block', 'Mode_of_Shipment', 'Product_importance', 'Gender']].head()) # 범주형 변수

x_train = pd.get_dummies(x_train, columns = ['Warehouse_block', 'Mode_of_Shipment'])
x_val = pd.get_dummies(x_val, columns = ['Warehouse_block', 'Mode_of_Shipment'])
x_test = pd.get_dummies(x_test, columns = ['Warehouse_block', 'Mode_of_Shipment'])
# x_train = x_train.drop(['Warehouse_block', 'Mode_of_Shipment'], axis=1)
print(x_train.columns)

## 파생 변수 생성
print(x_train['Product_importance'].value_counts())
x_train['Product_importance_new'] = x_train['Product_importance'].replace(2, 0)
x_val['Product_importance_new'] = x_val['Product_importance'].replace(2, 0)
x_test['Product_importance_new'] = x_test['Product_importance'].replace(2, 0)

x_train = x_train.drop('Product_importance', axis=1)
x_val = x_val.drop('Product_importance', axis=1)
x_test = x_test.drop('Product_importance', axis=1)

print(x_train.shape, x_val.shape, x_test.shape)

# 모델링
from xgboost import XGBClassifier
xgb_model = XGBClassifier(n_estimators=200, max_depth=10, random_state=555)
# print(help(XGBClassifier))
xgb_model.fit(x_train, y_train)

from sklearn.tree import DecisionTreeClassifier
dt_model = DecisionTreeClassifier(random_state=555)
dt_model.fit(x_train, y_train)

from sklearn.ensemble import RandomForestClassifier
rm_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=555)
rm_model.fit(x_train, y_train)

# 성능평가
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
print(xgb_model.predict(x_val))
print(roc_auc_score(xgb_model.predict(x_val), y_val))
print(roc_auc_score(dt_model.predict(x_val), y_val))
print(roc_auc_score(rm_model.predict(x_val), y_val))

x_test = x_test.reset_index().drop('index', axis=1)
df_pred = pd.DataFrame(rm_model.predict(x_test.drop('ID', axis=1)), columns=['Reached.on.Time_Y.N'])
print(df_pred)
print(x_test['ID'])
df_pred_submit = pd.concat([x_test['ID'], df_pred], axis=1)
print(df_pred_submit)

df_pred_submit.to_csv('./123456.csv', index=False)

print(roc_auc_score(rm_model.predict(x_test.drop('ID', axis=1)), y_test))








