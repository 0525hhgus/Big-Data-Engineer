import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 100)
df = pd.read_csv('basic1.csv')

df = df[(df['age'] >= 0) & (df['age'] - round(df['age']) == 0)]

# 구간 분할
df['range'] = pd.qcut(df['age'], q=3, labels=['group1','group2','group3'])

# 수량 비교
print(df['range'].value_counts())

# 중간이상 - 중간이하
g1_med = df[df['range'] == 'group1']['age'].median()
g2_med = df[df['range'] == 'group2']['age'].median()
g3_med = df[df['range'] == 'group3']['age'].median()

print(g1_med + g2_med + g3_med)