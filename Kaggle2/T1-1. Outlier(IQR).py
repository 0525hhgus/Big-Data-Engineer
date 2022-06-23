import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 100)

df = pd.read_csv('titanic.csv')

print(df['Fare'])

q1 = df['Fare'].quantile(0.25)
q3 = df['Fare'].quantile(0.75)
iqr = q3 - q1

outlier = df[(df['Fare'] < (q1 - 1.5*iqr)) | (df['Fare'] > (q3 + 1.5*iqr))]

print(outlier)

print(outlier[outlier['Sex'] == 'female'].shape[0])


