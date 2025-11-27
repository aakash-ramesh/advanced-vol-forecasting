import wrds
import pandas as pd
import os

# --- PATH CONFIGURATION ---
# Ensure the path ends with a slash or use os.path.join for safety
base_path = os.path.dirname(os.getcwd()) + "/advanced-vol-forecasting/data/master/"

# Create the directory if it doesn't exist (prevents save errors)
if not os.path.exists(base_path):
    os.makedirs(base_path)
    print(f"Created directory: {base_path}")

# --- CREDENTIALS ---
USERNAME = "aramesh342"
PASSWORD = "QuantSchool@1"

# Connect to WRDS
db = wrds.Connection(wrds_username=USERNAME, wrds_password=PASSWORD)

# --- CONFIGURATION ---
start_year = 2000
end_year = 2022
target_secids = (108105, 109820)  # SPX and SPY

# We define the prefixes for the tables we want
# 'vsurfd': Volatility Surface Daily
# 'opprcd': Option Prices Daily
# 'secprd': Security Prices Daily (Underlying price, Open, Close, Return, etc.)
# table_types = ['vsurfd', 'opprcd', 'secprd']

table_types = ['secprd']
print(f"Targeting SECIDs: {target_secids}")
print(f"Timeframe: {start_year} - {end_year}")
print(f"Saving to: {base_path}")

for year in range(start_year, end_year + 1):
    print(f"\n--- Processing Year: {year} ---")
    
    for t_type in table_types:
        try:
            # Construct table name (e.g., optionm_all.secprd2022)
            table_name = f"optionm_all.{t_type}{year}"
            
            # Construct the query
            query = f"""
                SELECT *
                FROM {table_name}
                WHERE secid IN {target_secids}
            """
            
            print(f"   Querying {table_name} ...")
            
            # Run Query
            df = db.raw_sql(query)
            
            if df.empty:
                print(f"   [!] No data found for {t_type} in {year}")
                continue

            # File naming: secprd_2022.parquet
            output_file = os.path.join(base_path, f"{t_type}_{year}.parquet")
            
            # Save
            df.to_parquet(output_file, index=False)
            
            rows = df.shape[0]
            size_mb = os.path.getsize(output_file) / (1024 * 1024)
            print(f"   ✔ Saved {os.path.basename(output_file)} ({rows:,} rows, {size_mb:.2f} MB)")

        except Exception as e:
            print(f"   [X] Error on {t_type} {year}: {e}")

db.close()
print("\nAll downloads complete.")