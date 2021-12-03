import pandas as pd

df_x = pd.read_csv('bike_x_train.csv', encoding='cp949')
df_y = pd.read_csv('bike_y_train.csv', encoding='cp949')
df_test = pd.read_csv('bike_x_test.csv', encoding='cp949')

df_y.columns = ['datetime', 'count']
print(df_x.info())
print(df_x['계절'].unique())
print(df_x['공휴일'].unique())
print(df_x['근무일'].unique())
print(df_x['날씨'].unique())

print(df_x.describe().T)

df = pd.concat([df_x, df_y], axis=1)
print(df.groupby(['계절'])['count'].sum())
print(df.groupby(['공휴일'])['count'].sum())
print(df.groupby(['근무일'])['count'].sum())
print(df.groupby(['날씨'])['count'].sum())

# 데이터 전처리
## 파생 변수 만들기
df_x['datetime'] = pd.to_datetime(df_x['datetime'])
df_x['year'] = df_x['datetime'].dt.year
print(df_x['year'].unique())
df_x['month'] = df_x['datetime'].dt.month
print(df_x['month'].unique())
df_x['day'] = df_x['datetime'].dt.day
print(df_x['day'].unique())
df_x['hour'] = df_x['datetime'].dt.hour
print(df_x['hour'].unique())
df_x['dayofweek'] = df_x['datetime'].dt.dayofweek
print(df_x['dayofweek'].unique())

df = pd.concat([df_x, df_y], axis=1)

print(df.groupby(['month'])['count'].sum())
df_x = df_x.drop(['month'], axis=1)
print(df.groupby(['day'])['count'].sum())
df_x = df_x.drop(['day'], axis=1)
print(df.groupby(['dayofweek'])['count'].sum())
df_x = df_x.drop(['dayofweek'], axis=1)

df_test['datetime'] = pd.to_datetime(df_test['datetime'])
df_test['year'] = df_test['datetime'].dt.year
df_test['hour'] = df_test['datetime'].dt.hour

df_x_id = df_x['datetime']
df_y_id = df_y['datetime']
df_test_id = df_test['datetime']
df_x = df_x.drop(['datetime'], axis=1)
df_y = df_y.drop(['datetime'], axis=1)
df_test = df_test.drop(['datetime'], axis=1)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=10)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

from xgboost import XGBRegressor

print(dir(XGBRegressor()))
model = XGBRegressor(n_estimators=200, max_depth=5, random_state=10)
model.fit(x_train, y_train)

y_pred = pd.DataFrame(model.predict(x_test)).rename(columns={0:'count'})
y_pred[y_pred['count'] < 0] = 0
print(y_pred)

from sklearn.metrics import r2_score
print(r2_score(y_test, y_pred))

y_pred_result = pd.DataFrame(model.predict(df_test)).rename(columns={0:'count'})
y_pred_result[y_pred_result['count'] < 0] = 0
print(y_pred_result)

result = pd.concat([df_test_id, y_pred_result], axis=1)
result.to_csv('222222.csv', index=False)
