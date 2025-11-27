import glob
import os
import pyarrow.parquet as pq
import pyarrow as pa

# Define the master directory
# ensuring the path ends in / so string concatenation works later
parent_dir = os.path.dirname(os.getcwd()) + "/advanced-vol-forecasting/data/master/"

def combine_parquet_files(file_prefix, output_filename):
    """
    Combines multiple yearly parquet files into one large file 
    without loading everything into RAM at once.
    """
    # 1. Find all files matching the prefix (e.g., "opprcd_*.parquet")
    # We sort them to ensure the data is ordered chronologically (2000, 2001, etc.)
    search_pattern = parent_dir + f"{file_prefix}_*.parquet"
    files = sorted(glob.glob(search_pattern))
    
    if not files:
        print(f"❌ No files found for pattern: {search_pattern}")
        return

    print(f"Found {len(files)} files for '{file_prefix}'. Starting merge...")

    # 2. Read the schema from the first file
    # We need to know the column structure to initialize the master file
    try:
        first_table = pq.read_table(files[0])
        schema = first_table.schema
    except Exception as e:
        print(f"CRITICAL ERROR reading first file {files[0]}: {e}")
        return

    # 3. Open a ParquetWriter
    # This acts like an open stream to the hard drive
    with pq.ParquetWriter(output_filename, schema, compression='snappy') as writer:
        
        # Write the first file we already loaded
        writer.write_table(first_table)
        print(f"   Processed: {os.path.basename(files[0])}")

        # 4. Iterate through the rest of the files
        for file in files[1:]:
            try:
                # Read specific yearly file
                table = pq.read_table(file)
                
                # Check if schemas match (crucial for 23 years of data)
                if table.schema != schema:
                    # Sometimes int vs float types change slightly over 20 years. 
                    # We cast to the original schema to be safe.
                    table = table.cast(schema)
                
                # Append to the master file
                writer.write_table(table)
                print(f"   Processed: {os.path.basename(file)}")
                
            except Exception as e:
                print(f"   ⚠️ Error merging {os.path.basename(file)}: {e}")

    # Get final file size
    if os.path.exists(output_filename):
        size_gb = os.path.getsize(output_filename) / (1024 * 1024 * 1024)
        print(f"✅ Successfully saved {output_filename} ({size_gb:.2f} GB)\n")
    else:
        print(f"❌ Failed to create {output_filename}\n")

# --- EXECUTE MERGE ---

print(f"Working directory: {parent_dir}\n")

# 1. Combine Volatility Surface
combine_parquet_files("vsurfd", parent_dir + "COMBINED_vsurfd_2000_2022.parquet")

# 2. Combine Security Prices (New Step)
combine_parquet_files("secprd", parent_dir + "COMBINED_secprd_2000_2022.parquet")

# 3. Combine Option Prices (Largest, do last)
combine_parquet_files("opprcd", parent_dir + "COMBINED_opprcd_2000_2022.parquet")