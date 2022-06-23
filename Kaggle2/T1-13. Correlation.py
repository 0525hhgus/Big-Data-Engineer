# 상관관계 구하기
# 주어진 데이터에서 상관관계를 구하고, quality와의 상관관계가 가장 큰 값과, 가장 작은 값을 구한 다음 더하시오!
# 단, quality와 quality 상관관계 제외, 소수점 둘째 자리까지 출력

import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 100)

df = pd.read_csv('winequality-red.csv')

print(df.head())

df_corr = df.corr()['quality']
df_corr = df_corr.drop('quality')

print(df_corr.max())
print(df_corr.min())

print(df_corr.max()+df_corr.min())
