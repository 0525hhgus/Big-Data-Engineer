# 주어진 데이터 셋에서 age컬럼 상위 20개의 데이터를 구한 다음
# f1의 결측치를 중앙값으로 채운다.
# 그리고 f4가 ISFJ와 f5가 20 이상인
# f1의 평균값을 출력하시오!
import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 100)

df = pd.read_csv('basic1.csv')

df_top20 = df.reset_index().sort_values('age', ascending=False).head(20)
print(df_top20)

df_top20['f1'] = df_top20['f1'].fillna(df_top20['f1'].median())
print(df_top20)

print(df_top20[(df_top20['f4'] == 'ISFJ') & (df_top20['f5'] >= 20)]['f1'].mean())