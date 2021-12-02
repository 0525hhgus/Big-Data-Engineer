import pandas as pd

df = pd.read_csv('boston.csv')
# print(df.shape)

col = df.columns
sum_unique = 0


for c in col:
   sum_unique += pd.DataFrame(df[c].unique()).count()[0]

# print(sum_unique)
# print(len(col))
print(sum_unique/len(col))
