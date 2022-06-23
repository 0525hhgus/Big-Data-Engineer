import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 100)

df = pd.read_csv('house.csv')

print(df['SalePrice'])

s1 = df['SalePrice'].skew() # 왜도
k1 = df['SalePrice'].kurt() # 첨도
print('before')
print(s1, k1)

df['SalePrice'] = np.log1p(df['SalePrice'])
s2 = df['SalePrice'].skew() # 왜도
k2 = df['SalePrice'].kurt() # 첨도
print('after')
print(s2, k2)

print(round(s1+s2+k1+k2, 2))
