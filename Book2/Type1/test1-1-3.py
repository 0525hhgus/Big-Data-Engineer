import pandas as pd
import numpy as np

df = pd.read_csv('https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/datasets/airquality.csv')
df = df.drop('Unnamed: 0', axis=1)

print(df.groupby('Month')['Temp'].mean())
