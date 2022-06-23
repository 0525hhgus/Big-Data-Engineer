import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 100)

df = pd.read_csv('basic1.csv')
df2 = df[df['f2'] == 1]['f1'].cumsum()
print(df2)

df2 = df2.fillna(method='bfill')
print(df2)
print(df2.mean())