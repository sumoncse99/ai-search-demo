import os
import pandas as pd
import json

input_folder = './products-2017/'
output_file = 'combined_products.parquet'

all_records = []

for subfolder in os.listdir(input_folder):
    subfolder_path = os.path.join(input_folder, subfolder)
    if os.path.isdir(subfolder_path):
        for filename in os.listdir(subfolder_path):
            if filename.endswith('.json'):
                filepath = os.path.join(subfolder_path, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    try:
                        data = json.load(f)
                        if isinstance(data, list):
                            all_records.extend(data)
                        elif isinstance(data, dict):
                            all_records.append(data)
                    except json.JSONDecodeError:
                        # Try line-by-line loading (NDJSON)
                        f.seek(0)
                        for line in f:
                            try:
                                record = json.loads(line)
                                all_records.append(record)
                            except:
                                continue

# Convert to DataFrame
df = pd.DataFrame(all_records)

# Preview column names
print("Sample columns:", df.columns.tolist()[:10])

# Save as Parquet
df.to_parquet(output_file)
print(f"Saved {len(df)} records to {output_file}")
