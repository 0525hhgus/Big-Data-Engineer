import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 100)

df = pd.read_csv('titanic.csv')

outlier = df[df['Age'] - df['Age'].round() != 0]
print(outlier)

ceil_age = np.mean(np.ceil(df['Age']))
floor_age = np.mean(np.floor(df['Age']))
trunc_age = np.mean(np.trunc(df['Age']))

print(ceil_age)
print(ceil_age + floor_age + trunc_age)
