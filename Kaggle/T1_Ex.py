import pandas as pd
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', 100)


df = pd.read_csv('mtcars.csv')
# print(df['qsec'])
print(df.shape[1])

scaler = MinMaxScaler()

df['qsec'] = scaler.fit_transform(df[['qsec']])

print(len(df[df['qsec'] > 0.5]))