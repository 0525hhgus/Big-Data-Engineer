import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 100)

df = pd.read_csv('basic1.csv')

print(df.isna().sum())

df_enfj = df[df['f4'] == 'ENFJ']
df_infp = df[df['f4'] == 'INFP']

print(np.abs(df_enfj['f1'].std() - df_infp['f1'].std()))

