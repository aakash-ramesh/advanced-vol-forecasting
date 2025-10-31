import pandas as pd
parent_dir = "/Users/aramesh/Documents/GitHub/advanced-vol-forecasting/"
data = pd.read_parquet("data/raw/opprcd2022_spy.parquet")
print(data.head())