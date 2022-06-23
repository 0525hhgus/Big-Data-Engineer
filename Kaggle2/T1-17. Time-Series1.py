import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 100)
df = pd.read_csv('basic2.csv')

print(df.head())
df['Date'] = pd.to_datetime(df['Date'])
df['year'] = df['Date'].dt.year
df['month'] = df['Date'].dt.month
df['day'] = df['Date'].dt.day

print(df[(df['year'] == 2022) & (df['month'] == 5)]['Sales'].median())
