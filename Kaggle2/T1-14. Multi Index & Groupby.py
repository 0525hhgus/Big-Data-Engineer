# city와 f4를 기준으로 f5의 평균값을 구한 다음, f5를 기준으로 상위 7개 값을 모두 더해 출력하시오 (소수점 둘째자리까지 출력)
import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 100)

df = pd.read_csv('basic1.csv')
print(df.head())

df_group = df.groupby(['city', 'f4']).mean()['f5']
print(df_group)

df_top7 = df.reset_index().sort_values('f5', ascending=False).head(7)
print(df_top7['f5'].sum())