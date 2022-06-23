# 주어진 데이터 셋에서 f2가 0값인 데이터를 age를 기준으로 오름차순 정렬하고
# 앞에서 부터 20개의 데이터를 추출한 후
# f1 결측치(최소값)를 채우기 전과 후의 분산 차이를 계산하시오 (소수점 둘째 자리까지)

import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 100)
df = pd.read_csv('basic1.csv')

df_var = df[df['f2'] == 0].reset_index().sort_values('age', ascending=True).head(20)
var1 = df_var['f1'].var()
print(df_var)
print(var1)

df_var['f1'] = df_var['f1'].fillna(df_var['f1'].min())
var2 = df_var['f1'].var()
print(df_var)
print(var2)

print(var1-var2)




