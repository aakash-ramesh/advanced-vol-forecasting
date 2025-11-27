import os
import requests
import numpy as np
import pandas as pd
import joblib
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# --- CONFIGURATION ---
# Use a long history to learn regimes properly
START_DATE = "1970-01-01" 
END_DATE   = "2023-01-01" 
FRED_API_KEY = "0edd0176511cb5f2acbd2333dcbe28b1" # Replace with your key if needed

# Define output path
base_path = "data/master/"
if not os.path.exists(base_path): os.makedirs(base_path)

# --- 1. DATA FETCHING (FRED) ---
SERIES_IDS = [
    "GS10",       # 10-Year Treasury
    "TB3MS",      # 3-Month T-Bill
    "BAA",        # Baa Corporate Bond
    "AAA",        # Aaa Corporate Bond
    "INDPRO",     # Industrial Production
    "CPIAUCSL",   # CPI
    "UNRATE",     # Unemployment
    "USRECM"      # NBER Recession Indicator (for validation plot only)
]

def fetch_fred_series(series_id):
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id, "api_key": FRED_API_KEY,
        "file_type": "json", "observation_start": START_DATE, "observation_end": END_DATE
    }
    r = requests.get(url, params=params)
    r.raise_for_status()
    data = r.json()["observations"]
    df = pd.DataFrame(data)
    df['value'] = pd.to_numeric(df['value'].replace('.', np.nan), errors='coerce')
    df['date'] = pd.to_datetime(df['date']) + pd.offsets.MonthEnd(0)
    return df.set_index('date')['value'].rename(series_id)

print("Fetching FRED data...")
df_list = [fetch_fred_series(s) for s in SERIES_IDS]
df = pd.concat(df_list, axis=1).dropna(how='all').sort_index()

# --- 2. FEATURE ENGINEERING ---
# Create stationarity and meaningful economic signals
df["TERM_SPREAD"]   = df["GS10"] - df["TB3MS"]
df["CREDIT_SPREAD"] = df["BAA"]  - df["AAA"]
df["INDPRO_YoY"]    = pd.Series(np.log(df["INDPRO"])).diff(12) * 100
df["CPI_YoY"]       = pd.Series(np.log(df["CPIAUCSL"])).diff(12) * 100
df["UNRATE_Delta"]  = df["UNRATE"].diff(6) # 6-month change in unemployment

# Select features for HMM
features = ["TERM_SPREAD", "CREDIT_SPREAD", "INDPRO_YoY", "CPI_YoY", "UNRATE_Delta"]
df_model = df[features].dropna()

print(f"Model Data Range: {df_model.index.min()} to {df_model.index.max()}")

# --- 3. MODEL TRAINING ---
X = df_model.values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train HMM with 3 states (Low, Neutral, High Risk)
print("Training HMM...")
model = GaussianHMM(n_components=3, covariance_type="full", n_iter=1000, random_state=42)
model.fit(X_scaled)

# --- 4. STATE SORTING (CRITICAL) ---
# HMM labels (0,1,2) are random. We must sort them logically.
# We assume "High Risk" (Crisis) has the highest CREDIT SPREAD.
# We assume "Low Risk" (Expansion) has the lowest CREDIT SPREAD.

# Get the mean Credit Spread (index 1 in X) for each state
means = model.means_
credit_spread_idx = features.index("CREDIT_SPREAD")
state_means = means[:, credit_spread_idx]

# Create a mapping: 0 -> Lowest Risk, 2 -> Highest Risk
sorted_indices = np.argsort(state_means) # e.g., [2, 0, 1]
state_map = {old: new for new, old in enumerate(sorted_indices)}

print(f"State Mapping (based on Credit Spread): {state_map}")
# 0 = Low Vol/Expansion, 1 = Neutral, 2 = High Vol/Crisis

# --- 5. PREDICTION & SAVING ---
hidden_states = model.predict(X_scaled)
filtered_probs = model.predict_proba(X_scaled) # Prob of being in state k given data 0...t

# Apply mapping
df_model["hmm_state"] = [state_map[s] for s in hidden_states]
for i in range(3):
    # Map the probability columns to the sorted logic
    mapped_idx = state_map[i] # Where did old state 'i' go?
    # Note: We need to assign the probability OF the old state 'i' TO the column 'prob_state_{mapped_idx}'
    # But strictly speaking, we want the columns to be ordered 0 (Low), 1 (Med), 2 (High).
    
    # Let's create a temporary array to hold sorted probs
    pass

# Re-organize probabilities into sorted columns
sorted_probs = np.zeros_like(filtered_probs)
for old_idx, new_idx in state_map.items():
    sorted_probs[:, new_idx] = filtered_probs[:, old_idx]

df_model["prob_low_risk"]  = sorted_probs[:, 0]
df_model["prob_med_risk"]  = sorted_probs[:, 1]
df_model["prob_high_risk"] = sorted_probs[:, 2]

# Save for the next step
out_path = base_path + "hmm_macro_regimes_monthly.csv"
df_model.to_csv(out_path)
print(f"Saved monthly regimes to {out_path}")

# Save the model
joblib.dump(model, base_path + "hmm_model_trained.pkl")
joblib.dump(scaler, base_path + "hmm_scaler.pkl")

# --- 6. VALIDATION PLOT ---
# Compare HMM High Risk Prob vs NBER Recessions
plt.figure(figsize=(12, 6))
plt.plot(df_model.index, df_model["prob_high_risk"], label="HMM High Risk Prob", color='black', linewidth=1)
plt.fill_between(df.index, 0, df["USRECM"].reindex(df.index), color='red', alpha=0.3, label="NBER Recession")
plt.title("HMM Regime Probability vs. NBER Recessions (1970-2022)")
plt.legend()
plt.show()