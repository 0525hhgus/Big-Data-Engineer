import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 100)
df = pd.read_csv('basic2.csv')

#1일 차이가 나는 시차 특성 만들기
df['previous_PV'] = df['PV'].shift(1)
df.head()

# 1일 씩 미뤘음으로 가장 앞이 결측값이 됨 (바로 뒤의 값으로 채움)
df['previous_PV'] = df['previous_PV'].fillna(method = 'bfill')
df.head()

# 조건에 맞는 1일 이전 PV의 합
cond = (df['Events'] == 1) & (df['Sales'] <= 1000000)
print(df[cond]['previous_PV'].sum())
