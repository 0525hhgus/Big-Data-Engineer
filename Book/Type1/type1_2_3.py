import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('boston.csv')
# print(df.shape)

scaler = MinMaxScaler()

df['MEDV'] = scaler.fit_transform(df[['MEDV']])
df_medv = df[df['MEDV'] > 0.5]['MEDV']

print(df_medv.count())