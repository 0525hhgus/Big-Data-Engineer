import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 100)
df1 = pd.read_csv('basic1.csv')
df2 = pd.read_csv('basic3.csv')

print(df1.head())
print(df2.head())

df_merge = pd.merge(df1, df2, how='left', on='f4')
print(df_merge.head())

df_merge = df_merge.dropna(subset=['r2'])
print(df_merge.head())

df_20 = df_merge.head(20)

print(df_20['f2'].sum())
