import pandas as pd

pd.set_option('display.max_columns', 100)

df = pd.read_csv('basic1.csv')
# print(df['age'].unique())
df = df[(df['age'] > 0) & (df['age']*10//10 == df['age'])]
# df = df[(df['age'] > 0) & (df['age']*10%10 == 0)]
# df = df[(df['age'] >= 0) & (round(df['age'],0) == df['age'])]


# print(df['age'].unique())

# print(df.shape)
df1 = df[:30]
df2 = df[30:60]
df3 = df[60:]
print(df1.shape)
print(df2.shape)
print(df3.shape)

print(df1['age'].median()+df2['age'].median()+df3['age'].median())