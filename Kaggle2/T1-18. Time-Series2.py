import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 100)
df = pd.read_csv('basic2.csv')

df['Date'] = pd.to_datetime(df['Date'])
df['year'] = df['Date'].dt.year
df['month'] = df['Date'].dt.month
df['day'] = df['Date'].dt.day
df['dayofweek'] = df['Date'].dt.dayofweek

df['weekend'] = df['dayofweek'].apply(lambda x: x>=5, 1, 0)
print(df[df['weekend'] == 1]['Sales'].mean() - df[df['weekend'] == 0]['Sales'].mean())