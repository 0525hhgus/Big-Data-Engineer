import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 100)

df = pd.read_csv('covid-vaccination-vs-death_ratio.csv')

print(df.head())

df_country = df.groupby('country').max()
df_country = df_country.sort_values(by='ratio', ascending=False)

print(df_country)

top = df_country['ratio'].head(10).mean()
bottom = df_country['ratio'].tail(10).mean()

print(round(top-bottom, 2))
