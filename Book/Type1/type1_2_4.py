import pandas as pd
df = pd.read_csv('boston.csv')
# print(df.shape)

df['AGE'] = round(df['AGE'], 0)

df_age = pd.DataFrame(df.groupby(['AGE'])['AGE'].count())
df_age.columns = ['CNT']
df_age = df_age.reset_index(drop=False)
df_age_sort = df_age.sort_values(by='CNT', ascending=False)

print(df_age_sort.iloc[0,0], df_age_sort.iloc[0,1])