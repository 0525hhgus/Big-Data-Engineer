import pandas as pd

df = pd.read_csv("boston.csv")

print(df.sort_values(by="MEDV", ascending=True)["MEDV"].head(10))