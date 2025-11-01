import pandas as pd
import numpy as np


VSURF_PATH = "data/raw/volsrfc2022_spx.parquet"     
SECPRC_PATH = "data/raw/secprc2022_spyx.parquet"    

vsurf = pd.read_parquet(VSURF_PATH)
secprc = pd.read_parquet(SECPRC_PATH)

vsurf.columns = [c.lower() for c in vsurf.columns]
secprc.columns = [c.lower() for c in secprc.columns]
vsurf["date"] = pd.to_datetime(vsurf["date"])
secprc["date"] = pd.to_datetime(secprc["date"])

need = ["date","days","delta","impl_volatility"]
missing = [c for c in need if c not in vsurf.columns]
if missing:
    raise ValueError(f"Missing columns in vsurf: {missing}")

vsurf["days_round"] = vsurf["days"].round().astype(int)

available_days = sorted(vsurf["days_round"].unique().tolist())
# e.g. [10, 30, 60, 91, 122, 152, 182, 273, 365, 547, 730]

# target maturities we conceptually want
desired_days = [9, 30, 60]   # we’ll map 9 -> nearest available (10)
def nearest(target, available):
    return min(available, key=lambda x: abs(x - target))

D1  = nearest(9, available_days)   
D30 = nearest(30, available_days)  
D60 = nearest(60, available_days) 
desired_deltas = [-50.0, +50.0, -25.0, +25.0, +75.0]

all_deltas = sorted(vsurf["delta"].unique().tolist())  # signed percent deltas

iv_grid = vsurf.pivot_table(index="date",
                            columns=["days_round","delta"],
                            values="impl_volatility",
                            aggfunc="mean").sort_index()

def get_iv(days_target, delta_target):
    d_actual = nearest(days_target, available_days)
    x_actual = nearest(delta_target, all_deltas)
    key = (d_actual, x_actual)
    if key not in iv_grid.columns:
        # Return empty series if truly missing; the nearest match logic already applied
        return pd.Series(index=iv_grid.index, dtype=float)
    name = f"iv_d{d_actual}_d{int(x_actual)}"
    return iv_grid[key].rename(name)


def atm_iv(days_target):
    c50 = get_iv(days_target, +50.0)
    p50 = get_iv(days_target, -50.0)
    if c50.notna().any() and p50.notna().any():
        s = pd.concat([c50, p50], axis=1).mean(axis=1)
    else:
        s = c50 if c50.notna().any() else p50
    s.name = f"atm_iv_{nearest(days_target, available_days)}d"
    return s


feat = pd.DataFrame(index=iv_grid.index)


feat[f"atm_iv_{D1}d"]  = atm_iv(D1)
feat[f"atm_iv_{D30}d"] = atm_iv(D30)


feat["term_slope"] = feat[f"atm_iv_{D30}d"] - feat[f"atm_iv_{D1}d"]

# 25Δ risk reversal at D30: IV(-25) - IV(+25)
put25  = get_iv(D30, -25.0)
call25 = get_iv(D30, +25.0)
feat["rr_25d"] = put25 - call25

# 25Δ butterfly at D30: 0.5*(IV(-25)+IV(+25)) - ATM
mid_wings = pd.concat([put25, call25], axis=1).mean(axis=1)
feat["bfly_25d"] = mid_wings - feat[f"atm_iv_{D30}d"]

# Smile slope (call wing) at D30: IV(+75) - ATM
call75 = get_iv(D30, +75.0)
feat["smile_slope"] = call75 - feat[f"atm_iv_{D30}d"]

# Term curvature across (D60, D30, D1): IV(D60) - 2*IV(D30) + IV(D1)
feat[f"atm_iv_{D60}d"] = atm_iv(D60)
feat["term_curv"] = feat[f"atm_iv_{D60}d"] - 2*feat[f"atm_iv_{D30}d"] + feat[f"atm_iv_{D1}d"]

secprc = pd.read_parquet(SECPRC_PATH)
secprc.columns = [c.lower() for c in secprc.columns]
secprc["date"] = pd.to_datetime(secprc["date"])

df = secprc.copy()
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

O = df["open"].astype(float)
H = df["high"].astype(float)
L = df["low"].astype(float)
C = df["close"].astype(float)

rs_var = (np.log(H / O) * np.log(H / C)) + (np.log(L / O) * np.log(L / C))

window = 30
annual_factor = 252 / window
rs_var = pd.Series(rs_var, index=df.index)

df["rv30_ann"] = annual_factor * rs_var.rolling(window).sum()

feat = feat.merge(df[["date", "rv30_ann"]], on="date", how="inner")
feat["ivvar_30d"] = feat["atm_iv_30d"] ** 2
feat["vrp_30d"] = feat["ivvar_30d"] - feat["rv30_ann"]


out_path = f"data/master/iv_features_spx_{D1}_{D30}_{D60}.parquet"  # e.g., iv_features_spx_10_30_60.parquet
feat.reset_index().to_parquet(out_path, index=False)
vix_feat = pd.read_csv("data/master/vix_all_2022.csv")
vix_feat["date"] = pd.to_datetime(vix_feat["date"])
close_cols = [c for c in vix_feat.columns if c.endswith("_close")]
feat = feat.merge(vix_feat, on="date", how="inner")

for c in close_cols:
    base = c[:-6]  # strip "_close"
    lvl = feat[c]
    feat[f"{base}_ret1"]  = np.log(lvl / lvl.shift(1))
    feat[f"{base}_ma5"]   = lvl.rolling(5).mean()
    feat[f"{base}_ma20"]  = lvl.rolling(20).mean()

# VIX-based cross features if vix_close is present
vix_sigma = feat["vix_close"] / 100.0                 # convert % to decimal
feat["vix_var"] = vix_sigma ** 2

feat["vrp_30d_vix"]   = feat["vix_var"] - feat["rv30_ann"]
feat["ivrv_ratio_vix"] = feat["vix_var"] / feat["rv30_ann"]

feat["vix_minus_atmiv30"]  = vix_sigma - feat["atm_iv_30d"]
feat["atmiv_to_vix_ratio"] = feat["atm_iv_30d"] / vix_sigma

feat = feat.drop_duplicates(subset=["date"], keep="last")

print("Saved:", out_path)
print("Columns:", feat.columns.tolist())
print(feat.head(45))

feat.to_csv("data/master/iv_features_spx_2022.csv", index=False)
import pandas as pd
