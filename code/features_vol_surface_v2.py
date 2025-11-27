import pandas as pd
import numpy as np
import os

parent_dir = os.path.dirname(os.getcwd())+"/advanced-vol-forecasting/data/master/"

# --- CONFIGURATION ---
# Use the combined files generated in the previous step
VSURF_PATH = parent_dir+"COMBINED_vsurfd_2000_2022.parquet"
SECPRC_PATH = parent_dir+"COMBINED_secprd_2000_2022.parquet" # Assumes you combined secprc similarly, or use the raw file logic
VIX_PATH = parent_dir+"vix_extended.csv" # Update this path if you have this file

TARGET_SECID = 108105 # SPX (Index). Change to 109820 for SPY.

print("Loading Data...")
vsurf = pd.read_parquet(VSURF_PATH)
secprc = pd.read_parquet(SECPRC_PATH)

# --- 1. PREPROCESSING & FILTERING ---
vsurf.columns = [c.lower() for c in vsurf.columns]
secprc.columns = [c.lower() for c in secprc.columns]

# FILTER: We must isolate the specific asset (SPX) from the combined file
vsurf = vsurf[vsurf['secid'] == TARGET_SECID].copy()
secprc = secprc[secprc['secid'] == TARGET_SECID].copy()

vsurf["date"] = pd.to_datetime(vsurf["date"])
secprc["date"] = pd.to_datetime(secprc["date"])

print(f"Data Loaded. Processing {len(vsurf)} surface rows and {len(secprc)} price rows for SECID {TARGET_SECID}...")

# Validation
need = ["date","days","delta","impl_volatility"]
missing = [c for c in need if c not in vsurf.columns]
if missing:
    raise ValueError(f"Missing columns in vsurf: {missing}")

# --- 2. SURFACE PIVOT ---
vsurf["days_round"] = vsurf["days"].round().astype(int)
available_days = sorted(vsurf["days_round"].unique().tolist())

# Helper to find nearest available maturity/delta
def nearest(target, available):
    return min(available, key=lambda x: abs(x - target))

D1  = nearest(9, available_days)   
D30 = nearest(30, available_days)  
D60 = nearest(60, available_days) 
all_deltas = sorted(vsurf["delta"].unique().tolist())

print(f"Mapped Target Maturities: 9d->{D1}d, 30d->{D30}d, 60d->{D60}d")

# Create the Master Grid (Date x [Days, Delta])
iv_grid = vsurf.pivot_table(index="date",
                            columns=["days_round","delta"],
                            values="impl_volatility",
                            aggfunc="mean").sort_index()

# Helper to extract IV from grid safely
def get_iv(days_target, delta_target):
    d_actual = nearest(days_target, available_days)
    x_actual = nearest(delta_target, all_deltas)
    key = (d_actual, x_actual)
    if key not in iv_grid.columns:
        return pd.Series(index=iv_grid.index, dtype=float)
    name = f"iv_d{d_actual}_d{int(x_actual)}"
    return iv_grid[key].rename(name)

def atm_iv(days_target):
    c50 = get_iv(days_target, +50.0)
    p50 = get_iv(days_target, -50.0)
    # Average Call and Put ATM IV
    if c50.notna().any() and p50.notna().any():
        s = pd.concat([c50, p50], axis=1).mean(axis=1)
    else:
        s = c50 if c50.notna().any() else p50
    s.name = f"atm_iv_{nearest(days_target, available_days)}d"
    return s

# --- 3. VOLATILITY FEATURES ---
feat = pd.DataFrame(index=iv_grid.index)

# Level
feat[f"atm_iv_{D1}d"]  = atm_iv(D1)
feat[f"atm_iv_{D30}d"] = atm_iv(D30)
feat[f"atm_iv_{D60}d"] = atm_iv(D60)

# Term Structure
feat["term_slope"] = feat[f"atm_iv_{D30}d"] - feat[f"atm_iv_{D1}d"]
feat["term_curv"] = feat[f"atm_iv_{D60}d"] - 2*feat[f"atm_iv_{D30}d"] + feat[f"atm_iv_{D1}d"]

# Skew / Smile (Risk Reversal & Butterfly)
put25  = get_iv(D30, -25.0)
call25 = get_iv(D30, +25.0)
call75 = get_iv(D30, +75.0)

feat["rr_25d"] = put25 - call25
mid_wings = pd.concat([put25, call25], axis=1).mean(axis=1)
feat["bfly_25d"] = mid_wings - feat[f"atm_iv_{D30}d"]
feat["smile_slope"] = call75 - feat[f"atm_iv_{D30}d"]

# --- 4. REALIZED VOLATILITY ---
# We use the already filtered 'secprc' dataframe
df = secprc.sort_values("date").set_index("date")

# Ensure floats
O = df["open"].astype(float)
H = df["high"].astype(float)
L = df["low"].astype(float)
C = df["close"].astype(float)

# Rogers-Satchell Variance
rs_var = (np.log(H / O) * np.log(H / C)) + (np.log(L / O) * np.log(L / C))

window = 30
annual_factor = 252 / window
rs_var = pd.Series(rs_var, index=df.index)

# Calculate Rolling RV
rv_series = annual_factor * rs_var.rolling(window).sum()
feat = feat.join(rv_series.rename("rv30_ann"), how="inner")

# Variance Risk Premium
feat["ivvar_30d"] = feat["atm_iv_30d"] ** 2
feat["vrp_30d"] = feat["ivvar_30d"] - feat["rv30_ann"]

# --- 5. OPTIONAL: VIX FEATURES ---
# This block handles the case where you might not have the 2000-2022 VIX file yet
if os.path.exists(VIX_PATH):
    print(f"Merging VIX data from {VIX_PATH}...")
    vix_feat = pd.read_csv(VIX_PATH)
    vix_feat["date"] = pd.to_datetime(vix_feat["date"])
    
    # Merge
    feat = feat.merge(vix_feat, on="date", how="inner")
    
    # Calculate Returns / MAs if columns exist
    close_cols = [c for c in vix_feat.columns if c.endswith("_close")]
    for c in close_cols:
        base = c[:-6]
        lvl = feat[c]
        feat[f"{base}_ret1"]  = np.log(lvl / lvl.shift(1))
        feat[f"{base}_ma5"]   = lvl.rolling(5).mean()
        
    # VIX-Specific Derived Features
    if "vix_close" in feat.columns:
        vix_sigma = feat["vix_close"] / 100.0
        feat["vix_var"] = vix_sigma ** 2
        feat["vrp_30d_vix"] = feat["vix_var"] - feat["rv30_ann"]
        feat["ivrv_ratio_vix"] = feat["vix_var"] / feat["rv30_ann"]
        feat["vix_minus_atmiv30"]  = vix_sigma - feat["atm_iv_30d"]
else:
    print("⚠️ VIX file not found. Skipping VIX-specific feature generation.")

# --- 6. CLEANUP & SAVE ---
feat = feat.drop_duplicates(subset=["date"], keep="last")
# out_parquet = f"iv_features_spx_2000_2022.parquet"
out_csv = parent_dir+f"iv_features_spx_2000_2022.csv"

# feat.to_parquet(out_parquet, index=False)
feat.to_csv(out_csv, index=False)

print("\nProcessing Complete.")
# print(f"Saved parquet to: {out_parquet}")
print(f"Saved csv to:     {out_csv}")
print(f"Final Shape:      {feat.shape}")
print(feat[['date', 'atm_iv_30d', 'term_slope', 'rv30_ann', 'vrp_30d']].tail())