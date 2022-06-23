import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 100)

df = pd.read_csv('basic1.csv')

print(df.head())

print(df.isna().sum()/df.shape[0])

df.drop('f3', axis=1)

df_dropna = df.copy().dropna()
city_mean = df_dropna.groupby('city').mean()['f1']
print(city_mean)

df_fillna = df_dropna.fillna(df_dropna['city'].map({"서울": city_mean['서울'], "부산": city_mean['부산'], "경기": city_mean['경기']}))
print(df_fillna['f1'].mean())
