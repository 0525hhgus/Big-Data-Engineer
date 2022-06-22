import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 100)

df = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')

print(df.info())

df_num_col = df.select_dtypes('number').columns.values
print(df_num_col)

df_num = df[df_num_col].copy()
# print(df_num.head())

df_num_corr = df_num.corr(method='pearson')
print(df_num_corr)

corr_list = []

for c in df_num.columns:
    # print(df_num_corr.loc[df_num_corr[c] >= 0.9, c])
    df_corr = df_num_corr.loc[df_num_corr[c] >= 0.9, c]
    for i in range(len(df_corr)):
        if df_corr[i] != 1:
            corr_list.append(df_corr.name)

print(corr_list)

