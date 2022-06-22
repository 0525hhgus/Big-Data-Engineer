import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/datasets/airquality.csv')
df = df.drop('Unnamed: 0', axis=1)
'''
wind_mean = df['Wind'].mean()
print(wind_mean)

mms = MinMaxScaler()

df['Wind'] = mms.fit_transform(df[['Wind']])
wind_mean2 = df['Wind'].mean()
print(wind_mean2)

print(wind_mean-wind_mean2)
'''

Min = np.min(df['Wind'])
Max = np.max(df['Wind'])

df['min_max_wind'] = round((df['Wind']-Min)/(Max-Min), 2)
min_max_wind_mean = df['min_max_wind'].mean()
print(min_max_wind_mean)

Mean = np.mean(df['Wind'])
Std = np.std(df['Wind'])

df['z_wind'] = round((df['Wind']-Mean)/Std, 2)
z_wind_mean = df['z_wind'].mean()
print(z_wind_mean)

print(min_max_wind_mean-z_wind_mean)


