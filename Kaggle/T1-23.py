import pandas as pd

df = pd.read_csv('basic1.csv')

top10 = df['f1'].sort_values(ascending=False).iloc[9]
df['f1'] = df['f1'].fillna(top10)
old_median = df['f1'].median()

print(df.shape)
df = df.drop_duplicates(subset=['age'])
new_median = df['f1'].median()
print(df.shape)

print(old_median, new_median)
print(abs(old_median-new_median))