import wrds
import pandas as pd

USERNAME = ""
PASSWORD = ""

db = wrds.Connection(wrds_username=USERNAME, wrds_password=PASSWORD)

queries = {
    "opprcd2022_spy": """
        SELECT *
        FROM optionm_all.opprcd2022
        WHERE secid IN (109820)
    """,
    "volsrfc2022_spx": """
        SELECT *
        FROM optionm_all.vsurfd2022
        WHERE secid IN (108105)
    """,
    "secprc2022_spy": """
        SELECT *
        FROM optionm_all.secprd2022
        WHERE secid IN (109820,108105)
    """
}

for name, query in queries.items():
    print(f"\n Running query for {name} ...")
    df = db.raw_sql(query)
    print(f"   → Retrieved {df.shape[0]:,} rows and {df.shape[1]} columns")

    parquet_path = f"{name}.parquet"
    df.to_parquet(parquet_path, index=False)
    print(f" Saved to {parquet_path}")

db.close()
print("\nAll queries complete and saved.")
