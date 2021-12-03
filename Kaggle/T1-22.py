import pandas as pd

pd.set_option('display.max_columns', 100)

df = pd.read_csv('basic2.csv')

df['Date'] = pd.to_datetime(df['Date'])

df['year'] = df['Date'].dt.year
df['week'] = df['Date'].dt.week

df_w = pd.DataFrame(df.groupby(['year', 'week'])['Sales'].sum())

print(df_w['Sales'].max()-df_w['Sales'].min())