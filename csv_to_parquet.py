import pandas as pd
import glob
import os

csv_files = glob.glob("data/offline_csv/*.csv")

dfs = []
for file in csv_files:
    df = pd.read_csv(file)
    df["Company"] = os.path.basename(file).replace(".csv", "")
    dfs.append(df)

final_df = pd.concat(dfs, ignore_index=True)

output_path = "data/historical_prices_fixed.parquet"
final_df.to_parquet(output_path)

print("✅ Offline CSV converted to Parquet successfully")
print("Saved at:", output_path)
