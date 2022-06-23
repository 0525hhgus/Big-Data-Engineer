import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 100)

df = pd.read_csv('basic1.csv')

print(df.isna().sum())
df = df[~df['f1'].isnull()]
print(df.isna().sum())

df_group = df.groupby(['city', 'f1']).sum()
print(df_group)

# print(df_group.iloc[0])
print(df[(df['city'] == '경기') & (df['f2'] == 0)]['f1'])



