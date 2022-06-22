import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 100)

df = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')

print(df.head())

# df['encoded_attrition'] = df['Attrition'].replace('Yes', 1).replace('No', 0)
target_map = {'Yes':1, 'No':0}
df['encoded_attrition'] = df['Attrition'].apply(lambda x: target_map[x])

print(df['encoded_attrition'].value_counts())
