import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('boston.csv')
# print(df.shape)

scaler = StandardScaler()

df['DIS'] = scaler.fit_transform(df[['DIS']])
df_dis = df[(df['DIS'] > 0.4) & (df['DIS'] < 0.6)]

print(round(df_dis['DIS'].mean(), 2))

