import pandas as pd

df = pd.read_csv('boston.csv')
# print(df.shape)
df = df.drop(columns=['CHAS','RAD'])

# print(df.shape)
# print(df.describe())

df_desc = df.describe().iloc[[4,6]].T
# print(df_desc)

print(df_desc['75%']-df_desc['25%'])