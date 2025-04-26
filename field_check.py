import pandas as pd

df = pd.read_parquet('combined_products.parquet')
print(df.columns)
