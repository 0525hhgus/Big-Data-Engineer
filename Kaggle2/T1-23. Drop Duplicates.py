import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 100)
df = pd.read_csv('basic1.csv')

fill_value = df.reset_index().sort_values('f1', ascending=False).head(10)['f1'].iloc[9]
print(fill_value)

df['f1'] = df['f1'].fillna(fill_value)

# 중복 제거 전 중앙 값
result1 = df['f1'].median()
result1

df = df.drop_duplicates(subset=['age'])

# 중복 제거 후 중앙 값
result2 = df['f1'].median()
result2

# 차이 (절대값)
print(abs(result1 - result2))