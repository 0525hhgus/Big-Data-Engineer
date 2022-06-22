import pandas as pd
import numpy as np

df = pd.read_csv('https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/datasets/airquality.csv')
df = df.drop('Unnamed: 0', axis=1)

df_na = df.isna().sum()
print(df_na) # Ozone

ozone_mean = df['Ozone'].mean()
print(ozone_mean)

df_na_fill = df.copy()
df_na_fill['Ozone'].fillna(0, inplace=True)
print(df_na_fill.isna().sum())

ozone_mean2 = df_na_fill['Ozone'].mean()
print(ozone_mean2)

print(ozone_mean-ozone_mean2)


