import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 100)

df = pd.read_csv('basic1.csv')

from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
df['f5'] = mms.fit_transform(df[['f5']])
print(df['f5'])

upper = df['f5'].quantile(0.95)
lower = df['f5'].quantile(0.05)
print(upper+lower)
