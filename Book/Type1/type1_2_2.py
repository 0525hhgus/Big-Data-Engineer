import pandas as pd

df = pd.read_csv('boston.csv')
# print(df.shape)
df_tax_decs = df.sort_values(by='TAX', ascending=False)['TAX']
df_tax_decs = df_tax_decs.reset_index(drop=True)
df_tax_asc = df.sort_values(by='TAX', ascending=True)['TAX']
df_tax_asc = df_tax_asc.reset_index(drop=True)

# print(df_tax_decs)
# print(df_tax_asc)

df_tax = pd.concat([df_tax_asc, df_tax_decs], axis=1)
# print(df_tax)

df_tax['diff'] = abs(df_tax.iloc[:,0] - df_tax.iloc[:,1])
print(df_tax['diff'].var())
