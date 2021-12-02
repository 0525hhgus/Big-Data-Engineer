import pandas as pd

df = pd.read_csv('boston.csv')
# print(df.shape)
df_medv = df.sort_values(by=['MEDV'], ascending=False)['MEDV']
# print(df_medv.iloc[29])

df_medv.iloc[:28] = df_medv.iloc[29]
print(df_medv.mean(), df_medv.median(), df_medv.min(), df_medv.max())