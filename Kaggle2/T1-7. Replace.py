import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 100)

df = pd.read_csv('basic1.csv')

df['f4'] = df['f4'].replace('ESFJ', 'ISFJ')
print(df[(df['city'] == '경기') & (df['f4'] == 'ISFJ')]['age'].max())
