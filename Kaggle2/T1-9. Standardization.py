import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

pd.set_option('display.max_columns', 100)

df = pd.read_csv('basic1.csv')
sds = StandardScaler()
df['f5'] = sds.fit_transform(df[['f5']])

print(df['f5'].median())

