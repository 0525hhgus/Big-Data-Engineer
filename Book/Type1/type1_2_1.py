import pandas as pd

df = pd.read_csv('boston.csv')
# print(df.shape)
df_tax = df[df['TAX'] > df['TAX'].median()][['CHAS', 'RAD']]

# print(df_tax['CHAS'].unique())
# print(df_tax['RAD'].unique())

df_tax_group = pd.DataFrame(df_tax.groupby(['CHAS', 'RAD'])['RAD'].count())

df_tax_group.columns = ['COUNT']
print(df_tax_group)
