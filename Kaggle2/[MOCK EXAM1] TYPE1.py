# 1. 첫번째 데이터 부터 순서대로 50:50으로 데이터를 나누고, 앞에서 부터 50%의 데이터(이하, A그룹)는 'f1'컬럼을 A그룹의 중앙값으로 채우고,
# 뒤에서부터 50% 데이터(이하, B그룹)는 'f1'컬럼을 B그룹의 최대값으로 채운 후, A그룹과 B그룹의 표준편차 합을 구하시오
import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 100)

df = pd.read_csv('basic1.csv')
print(df.shape)

df1 = df.iloc[:50] # A그룹
df2 = df.iloc[50:] # B그룹
# print(df1.shape, df2.shape)
# print(df1.head())
# print(df1.isna().sum())

df1.loc[:, 'f1'] = df1['f1'].fillna(df1['f1'].median())
df2.loc[:, 'f1'] = df2['f1'].fillna(df2['f1'].max())
# print(df1.loc[:, 'f1'])
# print(df1['f1'].fillna(df1['f1'].median()))
# print(df1.head())
# print(df1.isna().sum())

print("정답1")
print(round(df1['f1'].std()+df2['f1'].std(), 1))

# 2. 'f4'컬럼을 기준 내림차순 정렬과 'f5'컬럼기준 오름차순 정렬을 순서대로 다중 조건 정렬하고나서
# 앞에서부터 10개의 데이터 중 'f5'컬럼의 최소값 찾고, 이 최소값으로 앞에서 부터 10개의 'f5'컬럼 데이터를 변경함. 그
# 리고 'f5'컬럼의 평균값을 계산함

df_sort = df.reset_index().sort_values(['f4', 'f5'], ascending=[False, True])

min_value = df_sort.head(10)['f5'].min()
print(min_value)

df_sort.iloc[:10, 7] = min_value
print(round(df['f5'].mean(),2))


# 3. 'age' 컬럼의 IQR방식을 이용한 이상치 수와 표준편차*1.5방식을 이용한 이상치 수 합을 구하시오
q1 = df['age'].quantile(0.25)
q3 = df['age'].quantile(0.75)
iqr = q3 - q1

df_iqr = df[(df['age'] < q1 - 1.5*iqr) | (df['age'] > q3 + 1.5*iqr)]
print(df_iqr['age'].shape[0])

std = df['age'].std()*1.5
mean = df['age'].mean()

df_std = df[(df['age'] < mean - std) | (df['age'] > mean + std)]
print(df_std['age'].shape[0])

print(df_iqr['age'].shape[0] + df_std['age'].shape[0])







