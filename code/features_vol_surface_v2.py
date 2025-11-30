import pandas as pd
import numpy as np
import os

# ==========================================
# CONFIGURATION
# ==========================================
# Explicitly setting the base path to match your file structure
project_root = "/Users/areya/Desktop/Project/advanced-vol-forecasting"

# Define Data Directories
raw_dir = os.path.join(project_root, "data", "raw")
master_dir = os.path.join(project_root, "data", "master")

# Input Paths
VSURF_PATH = os.path.join(raw_dir, "COMBINED_vsurfd_2000_2022.parquet")
SECPRC_PATH = os.path.join(raw_dir, "COMBINED_secprd_2000_2022.parquet")
OPPRCD_PATH = os.path.join(raw_dir, "COMBINED_opprcd_2000_2022.parquet")
VIX_PATH = os.path.join(raw_dir, "vix_extended.csv")

# Output Path
if not os.path.exists(master_dir):
    os.makedirs(master_dir)
OUT_CSV = os.path.join(master_dir, "iv_features_spx_2000_2022.csv")

TARGET_SECID = 108105  # SPX Index

print("--- Starting Feature Engineering Pipeline (Complete) ---")
print(f"Data Source: {raw_dir}")

# ==========================================
# 1. PRICE DATA & REALIZED VOLATILITY (Rogers-Satchell)
# ==========================================
print("1. Calculating Rogers-Satchell Realized Volatility...")

if not os.path.exists(SECPRC_PATH):
    raise FileNotFoundError(f"Price file not found: {SECPRC_PATH}")

secprc = pd.read_parquet(SECPRC_PATH)
secprc.columns = [c.lower() for c in secprc.columns]
secprc = secprc[secprc['secid'] == TARGET_SECID].copy()
secprc["date"] = pd.to_datetime(secprc["date"])
secprc = secprc.sort_values("date").set_index("date")
secprc = secprc[~secprc.index.duplicated(keep='last')]

# Rogers-Satchell Variance
H, L, O, C = secprc["high"], secprc["low"], secprc["open"], secprc["close"]
rs_term1 = np.log(H / C) * np.log(H / O)
rs_term2 = np.log(L / C) * np.log(L / O)
rs_var_daily = rs_term1 + rs_term2

# 30-Day Rolling Annualized RV
rv_30d = np.sqrt(rs_var_daily.rolling(window=21).mean() * 252)

# Initialize Master DataFrame
feat = pd.DataFrame(index=secprc.index)
feat["rv30_ann"] = rv_30d

# Target: Next Day's RV (Square root of annualized daily RS var)
feat["target_rv_next_day"] = np.sqrt(rs_var_daily.shift(-1) * 252)

# ==========================================
# 2. IV SURFACE FEATURES (Slope, Curvature, Skew)
# ==========================================
print("2. Constructing IV Surface Features...")

if not os.path.exists(VSURF_PATH):
    raise FileNotFoundError(f"Surface file not found: {VSURF_PATH}")

vsurf = pd.read_parquet(VSURF_PATH)
vsurf.columns = [c.lower() for c in vsurf.columns]
vsurf = vsurf[vsurf['secid'] == TARGET_SECID].copy()
vsurf["date"] = pd.to_datetime(vsurf["date"])
vsurf["days_round"] = vsurf["days"].round().astype(int)

# Create lookup grid: Date x (Days, Delta)
iv_grid = vsurf.pivot_table(
    index="date", columns=["days_round", "delta"], 
    values="impl_volatility", aggfunc="mean"
).sort_index()

def get_iv(df_grid, d, x):
    """Safely extract IV for specific day (d) and delta (x)"""
    if (d, x) in df_grid.columns: return df_grid[(d, x)]
    if (d, -x) in df_grid.columns: return df_grid[(d, -x)] # Handle negative delta notation
    return pd.Series(np.nan, index=df_grid.index)

# A. ATM Levels
feat["iv_30d"] = get_iv(iv_grid, 30, 50)
feat["iv_60d"] = get_iv(iv_grid, 60, 50)
feat["iv_90d"] = get_iv(iv_grid, 91, 50)

# Fill missing data (weekends/holidays)
feat = feat.ffill()

# B. Term Structure
feat["term_slope"] = feat["iv_60d"] - feat["iv_30d"] # Contango/Backwardation
feat["term_curvature"] = feat["iv_30d"] - 2*feat["iv_60d"] + feat["iv_90d"]

# C. Skew (Risk Reversal Proxy at 30D)
# IV(Put 25) - IV(Call 25)
iv_put_25 = get_iv(iv_grid, 30, -25)
iv_call_25 = get_iv(iv_grid, 30, 25)
feat["skew_30d"] = iv_put_25 - iv_call_25

# ==========================================
# 3. OPTION FLOW FEATURES (Gamma, PC Ratio, Vega)
# ==========================================
print("3. Processing Option Flow Features (Heavy Compute)...")

if os.path.exists(OPPRCD_PATH):
    # Read specific columns to save memory
    flow_cols = ["secid", "date", "cp_flag", "volume", "open_interest", "gamma", "vega"]
    try:
        flow = pd.read_parquet(OPPRCD_PATH, columns=flow_cols)
    except:
        flow = pd.read_parquet(OPPRCD_PATH)[flow_cols]

    # Filter for SPX
    flow = flow[flow["secid"] == TARGET_SECID].copy()
    flow["date"] = pd.to_datetime(flow["date"])
    
    # Aggregations
    # 1. Put/Call Ratios
    flow['is_put'] = flow['cp_flag'] == 'P'
    flow['is_call'] = flow['cp_flag'] == 'C'
    
    # Conditional sums for P/C ratio
    put_vol = flow[flow['is_put']].groupby('date')['volume'].sum()
    call_vol = flow[flow['is_call']].groupby('date')['volume'].sum()
    feat['pc_ratio_vol'] = put_vol / (call_vol + 1)
    
    put_oi = flow[flow['is_put']].groupby('date')['open_interest'].sum()
    call_oi = flow[flow['is_call']].groupby('date')['open_interest'].sum()
    feat['pc_ratio_oi'] = put_oi / (call_oi + 1)

    # 2. Net Gamma Exposure (GEX Proxy)
    feat["net_gamma"] = flow.groupby("date").apply(lambda x: np.sum(x["gamma"] * x["open_interest"]))

    # 3. Vega Flow (Smart Money Proxy)
    feat["vega_flow"] = flow.groupby("date").apply(lambda x: np.sum(x["vega"] * x["volume"]))
    
    # Forward fill flow data
    feat = feat.ffill()
    print("   Flow features calculated.")
else:
    print(f"   WARNING: Option Price file not found at {OPPRCD_PATH}. Flow features skipped.")

# ==========================================
# 4. VARIANCE RISK PREMIUM (VRP)
# ==========================================
print("4. Calculating VRP...")
# VRP = IV^2 - RV^2
feat["vrp_30d"] = (feat["iv_30d"]**2) - (feat["rv30_ann"]**2)

# ==========================================
# 5. MACRO / VIX FEATURES
# ==========================================
print("5. Merging Macro/VIX Data...")
if os.path.exists(VIX_PATH):
    vix_df = pd.read_csv(VIX_PATH, parse_dates=["date"]).set_index("date")
    feat = feat.join(vix_df[['vix_close']], how='left')
    feat['vix_level'] = feat['vix_close'] / 100.0
    feat['vix_ma5'] = feat['vix_level'].rolling(5).mean()
    feat['vix_gap'] = feat['vix_level'] - feat['iv_30d'] # Spread between VIX and SPX IV
else:
    print(f"   WARNING: VIX file not found at {VIX_PATH}. Skipping VIX features.")

# ==========================================
# 6. HAR LAGS & CLEANUP
# ==========================================
print("6. Finalizing Dataset...")
# HAR Components
feat['rv_d'] = feat['rv30_ann']
feat['rv_w'] = feat['rv30_ann'].rolling(5).mean()
feat['rv_m'] = feat['rv30_ann'].rolling(22).mean()

# Drop rows with any NaNs (Crucial fix for MissingDataError)
initial_len = len(feat)
feat.dropna(inplace=True)
dropped = initial_len - len(feat)

print(f"   Dropped {dropped} rows containing NaNs/Lags.")
print(f"   Final Shape: {feat.shape}")

# Final Save
feat.to_csv(OUT_CSV)
print(f"--- Feature Engineering Complete ---")
print(f"Saved to: {OUT_CSV}")