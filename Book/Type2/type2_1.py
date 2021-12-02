import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score

df_x = pd.read_csv('x_train.csv', encoding='cp949')
df_y = pd.read_csv('y_train.csv', encoding='cp949')
df_test = pd.read_csv('x_test.csv', encoding='cp949')

x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=10)

# 데이터 전처리
x_train_id = x_train['cust_id']
x_test_id = x_test['cust_id']
df_test_id = df_test['cust_id']
x_train = x_train.drop(['cust_id'], axis=1)
x_test = x_test.drop(['cust_id'], axis=1)
df_test = df_test.drop(['cust_id'], axis=1)

## 결측치 처리
print(x_train.isnull().sum())
x_train['환불금액'] = x_train['환불금액'].fillna(0)
x_test['환불금액'] = x_test['환불금액'].fillna(0)
df_test['환불금액'] = df_test['환불금액'].fillna(0)


## 범주형 변수 인코딩
print(x_train.info())
encoder = LabelEncoder()
x_train['주구매상품'] = encoder.fit_transform(x_train['주구매상품'])
x_test['주구매상품'] = encoder.fit_transform(x_test['주구매상품'])
df_test['주구매상품'] = encoder.fit_transform(df_test['주구매상품'])

x_train['주구매지점'] = encoder.fit_transform(x_train['주구매지점'])
x_test['주구매지점'] = encoder.fit_transform(x_test['주구매지점'])
df_test['주구매지점'] = encoder.fit_transform(df_test['주구매지점'])

## 파생변수 만들기
condition = x_train['환불금액'] > 0
x_train.loc[condition, '환불금액_new'] = 1
x_train.loc[~condition, '환불금액_new'] = 0
x_train = x_train.drop(['환불금액'], axis=1)

condition = x_test['환불금액'] > 0
x_test.loc[condition, '환불금액_new'] = 1
x_test.loc[~condition, '환불금액_new'] = 0
x_test = x_test.drop(['환불금액'], axis=1)

condition = df_test['환불금액'] > 0
df_test.loc[condition, '환불금액_new'] = 1
df_test.loc[~condition, '환불금액_new'] = 0
df_test = df_test.drop(['환불금액'], axis=1)

## 표준화 크기
scaler = RobustScaler()
x_train = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns)
x_test = pd.DataFrame(scaler.fit_transform(x_test), columns=x_test.columns)
df_test = pd.DataFrame(scaler.fit_transform(df_test), columns=df_test.columns)

## 상관관계
x_corr = x_train[['총구매액', '최대구매액', '환불금액_new']].corr()
# print(x_corr)
x_train = x_train.drop(['최대구매액'], axis=1)
x_test = x_test.drop(['최대구매액'], axis=1)
df_test = df_test.drop(['최대구매액'], axis=1)

print(x_train.corr())
print(y_train)
print(df_test)

# 모델
model = DecisionTreeClassifier(max_depth=10, criterion='entropy', random_state=10)
model.fit(x_train, y_train['gender'])

# 모델 성능 평가
y_test_predict = model.predict_proba(x_test)
y_test_predict = pd.DataFrame(y_test_predict)[1]
print(pd.DataFrame(y_test_predict)[1])
print(y_test_predict)
print(y_test['gender'])
print(roc_auc_score(y_test['gender'], y_test_predict))

# 결과 예측하기
df_test_result = pd.DataFrame(model.predict(df_test))
result = pd.concat([df_test_id, df_test_result], axis=1).rename(columns = {1:'gender'})
print(result)

result.to_csv('./123456.csv', index=False)