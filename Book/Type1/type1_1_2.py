import pandas as pd

df = pd.read_csv('boston.csv')

df_mean = df['RM'].copy()
df_drop = df['RM'].copy()

df_mean.fillna(df_mean.mean(), inplace=True)
mean_std = df_mean.std()
# print(df_mean.isnull().sum())

df_drop.dropna(inplace=True)
# print(df_drop.isnull().sum())
drop_std = df_drop.std()
# print(mean_std)
# print(drop_std)
print(abs(mean_std-drop_std))