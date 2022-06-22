import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 100)

df = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')

# print(df.head())

df.info()
# dtypes: int64(26), object(9)

cat_feat = df.select_dtypes('object', 'category').columns.values
df_cat = df[cat_feat].copy()
print(df_cat.nunique().sort_values())

df_cat = df_cat.drop(['Over18'], axis=1)
print(df_cat)
