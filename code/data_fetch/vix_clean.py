import pandas as pd
import numpy as np

VIX_CSV  = "VIX_History.csv"   # update if paths differ on your machine
VVIX_CSV = "VVIX_History.csv"
VIX9_CSV  = "VIX9D_History.csv" 

def _norm_cols(df):
    df = df.copy()
    df.columns = (
        df.columns
          .str.strip()
          .str.lower()
          .str.replace(" ", "_")
          .str.replace("-", "_")
    )
    return df

def _load_vix_like(path, prefix):
    """
    Robust loader for VIX/VVIX history CSVs.
    Expects columns like: Date, VIX Close / VVIX Close (optionally Open/High/Low).
    Returns columns:
      ['date', f'{prefix}_close', f'{prefix}_open', f'{prefix}_high', f'{prefix}_low'] (when available)
    """
    x = pd.read_csv(path)
    x = _norm_cols(x)

    # date column
    date_col = None
    for cand in ("date", "trade_date"):
        if cand in x.columns:
            date_col = cand
            break
    if date_col is None:
        raise ValueError(f"No date-like column found in {path}")

    x["date"] = pd.to_datetime(x[date_col], errors="coerce")
    x = x.dropna(subset=["date"]).sort_values("date")

    # close column (flexible)
    close_candidates = [f"{prefix}_close", "close", f"{prefix}close"]
    close_col = next((c for c in close_candidates if c in x.columns), None)
    if close_col is None:
        # try any column containing 'close' and prefix token
        close_col = next((c for c in x.columns if "close" in c and prefix in c), None)
    if close_col is None:
        raise ValueError(f"No close column found for {prefix.upper()} in {path}")

    # optional O/H/L
    def _maybe(colnames):
        for c in colnames:
            if c in x.columns:
                return c
        return None

    open_col = _maybe([f"{prefix}_open", "open", f"{prefix}open"])
    high_col = _maybe([f"{prefix}_high", "high", f"{prefix}high"])
    low_col  = _maybe([f"{prefix}_low",  "low",  f"{prefix}low"])

    keep = {"date": "date", close_col: f"{prefix}_close"}
    if open_col: keep[open_col] = f"{prefix}_open"
    if high_col: keep[high_col] = f"{prefix}_high"
    if low_col:  keep[low_col]  = f"{prefix}_low"

    y = x[list(keep.keys())].rename(columns=keep)
    # ensure numeric
    for c in y.columns:
        if c != "date":
            y[c] = pd.to_numeric(y[c], errors="coerce")
    return y

# Load
vix  = _load_vix_like(VIX_CSV,  prefix="vix")
vix_9d = _load_vix_like(VIX_CSV,  prefix="vix9d")
# vvix = _load_vix_like(VVIX_CSV, prefix="vvix")

# vvix = pd.read_csv(VVIX_CSV,header=None, names=['date', 'vvix'])
# vvix['date'] = pd.to_datetime(vvix['date'])

# Robust VVIX load -> columns: date, vvix
_raw = pd.read_csv(VVIX_CSV)
_raw = _norm_cols(_raw)
date_col = next((c for c in _raw.columns if "date" in c), _raw.columns[0])
_raw["date"] = pd.to_datetime(_raw[date_col], errors="coerce")
_raw = _raw.dropna(subset=["date"]).sort_values("date")

# pick preferred vvix column (explicit 'vvix' or first numeric column)
candidates = [c for c in _raw.columns if c != date_col]
preferred = next((c for c in candidates if c == "vvix" or "vvix" in c), None)
if preferred is None:
    preferred = next((c for c in candidates if pd.to_numeric(_raw[c], errors="coerce").notna().any()), None)
if preferred is None:
    raise ValueError(f"Could not find a value column in {VVIX_CSV}")

vvix = _raw[["date", preferred]].rename(columns={preferred: "vvix"})
vvix["vvix"] = pd.to_numeric(vvix["vvix"], errors="coerce")
vvix = vvix.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


vix_all = vix.merge(vix_9d, on="date", how="outer")
vix_all = vix_all.merge(vvix, on="date", how="outer")
vix_all = vix_all.sort_values("date").reset_index(drop=True)

vix_all_2022 = vix_all[vix_all["date"].dt.year == 2022].reset_index(drop=True)

vix_all_2022.to_csv("vix_all_2022.csv", index=False)