import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 100)
df = pd.read_csv('basic2.csv')

df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day

def event_sales(x):
    if x['Events'] == 1:
        x['Sales2'] = x['Sales']*0.8
    else:
        x['Sales2'] = x['Sales']
    return x

df = df.apply(lambda x: event_sales(x), axis=1)
df_2022 = df[df['Year'] == 2022]
df_2023 = df[df['Year'] == 2023]

df_group1 = df_2022.groupby(['Month'])['Sales'].sum()
df_group2 = df_2023.groupby(['Month'])['Sales'].sum()

print(np.abs(df_group1.max() - df_group2.max()))





