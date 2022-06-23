import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 100)

df = pd.read_csv('basic1.csv')
df_age20up = df[df['age']>=20]
print(df_age20up)

df_age20up['f1'] = df_age20up['f1'].fillna(df_age20up['f1'].mode()[0])
print(df_age20up.isna().sum())

from sklearn.preprocessing import power_transform
yeo_johnson = power_transform(df_age20up[['f1']], method='yeo-johnson', standardize=False)
box_cox = power_transform(df_age20up[['f1']], method='box-cox', standardize=False)
print(yeo_johnson)
print(box_cox)

print(round(np.abs(yeo_johnson-box_cox).sum(), 2))

