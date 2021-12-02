import pandas as pd

df = pd.read_csv('boston.csv')

mean_zn = df['ZN'].mean()
std_zn = df['ZN'].std()

df_zn = df[(df['ZN'] > mean_zn+1.5*std_zn) | (df['ZN'] < mean_zn-1.5*std_zn)]['ZN']

print(sum(df_zn))
