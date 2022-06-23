import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 100)
df = pd.read_csv('basic2.csv')

df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df = df.set_index('Date')

df_w = df.resample('W').sum()
print(df_w)

ma = df_w['Sales'].max()
mi = df_w['Sales'].min()
print(ma-mi)