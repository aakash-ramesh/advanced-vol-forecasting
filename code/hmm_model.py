import os
import requests
import joblib
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict, Any
from pathlib import Path

from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

os.environ["FRED_API_KEY"] = "0edd0176511cb5f2acbd2333dcbe28b1"

@dataclass
class HMMConfig:
    n_states: int = 3
    covariance_type: str = "full"
    n_iter: int = 500
    random_state: int = 42
    verbose: bool = False


class VolatilityHMM:
    """
    3-state Gaussian HMM on (log) realized volatility or variance.

    Typical usage:
    --------------
    df = pd.read_csv("features_SPY.csv", parse_dates=["date"]).set_index("date")
    df["log_rv"] = np.log(df["rv30_ann"])   # or your realized vol/var column

    hmm = VolatilityHMM(HMMConfig())
    hmm.fit(df["log_rv"])
    df_regimes = hmm.attach_regimes(df)
    hmm.save("models/hmm_spy.joblib")
    """

    def __init__(self, config: Optional[HMMConfig] = None):
        self.config = config or HMMConfig()
        self.model: Optional[GaussianHMM] = None
        self.scaler: Optional[StandardScaler] = None
        self.state_order_: Optional[np.ndarray] = None  # mapping old_state -> ordered_state
        self.fitted_: bool = False

    # -------------------------
    # Core fit / predict
    # -------------------------
    def fit(self, series: pd.Series) -> "VolatilityHMM":
        """
        Fit HMM on a 1D time series (e.g., log realized variance).
        series: pd.Series indexed by date.
        """
        series = series.dropna()
        if series.empty:
            raise ValueError("Input series is empty after dropping NaNs.")

        X = series.values.reshape(-1, 1)

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        hmm = GaussianHMM(
            n_components=self.config.n_states,
            covariance_type=self.config.covariance_type,
            n_iter=self.config.n_iter,
            random_state=self.config.random_state,
            verbose=self.config.verbose,
        )

        hmm.fit(X_scaled)
        self.model = hmm

        # Sort states by mean of emission (low vol -> high vol)
        state_means = hmm.means_.flatten()
        order = np.argsort(state_means)  # 0 = lowest vol state
        self.state_order_ = order

        self.fitted_ = True
        return self

    def _check_fitted(self):
        if not self.fitted_ or self.model is None or self.scaler is None:
            raise RuntimeError("VolatilityHMM is not fitted yet. Call .fit() first.")

    def _transform_input(self, series: pd.Series) -> np.ndarray:
        series = series.astype(float)
        X = series.values.reshape(-1, 1)
        X_scaled = self.scaler.transform(X)
        return X_scaled

    def predict_states(self, series: pd.Series) -> pd.Series:
        """
        Return ordered state labels (0 = low vol, 1 = mid, 2 = high).
        """
        self._check_fitted()
        X_scaled = self._transform_input(series)
        raw_hidden = self.model.predict(X_scaled)

        # Map raw states -> ordered states
        mapping = {raw: ordered for ordered, raw in enumerate(self.state_order_)}
        ordered_states = np.array([mapping[s] for s in raw_hidden])
        return pd.Series(ordered_states, index=series.index, name="hmm_state")

    def predict_proba(self, series: pd.Series) -> pd.DataFrame:
        """
        Return ordered state probabilities with columns:
        ['prob_state_0', 'prob_state_1', 'prob_state_2'].
        """
        self._check_fitted()
        X_scaled = self._transform_input(series)
        raw_probs = self.model.predict_proba(X_scaled)  # shape (T, n_states)

        # raw probs columns correspond to raw states 0..K-1
        # reorder columns to match low -> high vol order
        ordered_probs = raw_probs[:, self.state_order_]

        cols = [f"prob_state_{k}" for k in range(self.config.n_states)]
        return pd.DataFrame(ordered_probs, index=series.index, columns=cols)

    # -------------------------
    # Regime-conditional means
    # -------------------------
    def regime_means(self) -> Dict[int, float]:
        """
        Regime-conditional expected value (in original scale, not standardized)
        for each ordered state k.
        """
        self._check_fitted()
        # model.means_ is in scaled space; invert using scaler
        means_scaled = self.model.means_.reshape(-1, 1)  # (K,1)
        means_original = self.scaler.inverse_transform(means_scaled).flatten()
        # Reorder according to state_order_
        ordered_means = means_original[self.state_order_]

        return {k: ordered_means[k] for k in range(self.config.n_states)}

    def regime_vol_forecast(self, series: pd.Series) -> pd.Series:
        """
        Simple example: For each date, forecast next-period log-vol as
        probability-weighted regime mean.
        (You can exponentiate if you want vol instead of log-vol.)
        """
        self._check_fitted()
        probs = self.predict_proba(series)
        means_dict = self.regime_means()
        means_vec = np.array([means_dict[k] for k in range(self.config.n_states)])

        # expected log-vol = sum_k p_k * mean_k
        exp_log_vol = probs.values @ means_vec
        return pd.Series(exp_log_vol, index=series.index, name="hmm_exp_log_vol")

    # -------------------------
    # Convenience: attach to DF
    # -------------------------
    def attach_regimes(
        self,
        df: pd.DataFrame,
        vol_col: str = "log_rv",
        prefix: str = "hmm_",
    ) -> pd.DataFrame:
        """
        Attach 'hmm_state' and 'prob_state_k' columns to df.
        vol_col: column in df on which the HMM was / will be fit (e.g., 'log_rv').
        """
        if vol_col not in df.columns:
            raise KeyError(f"{vol_col} not in DataFrame columns.")

        series = df[vol_col].dropna()
        states = self.predict_states(series)
        probs = self.predict_proba(series)
        exp_log_vol = self.regime_vol_forecast(series)

        out = df.copy()
        out.loc[states.index, prefix + "state"] = states.values
        for col in probs.columns:
            out.loc[probs.index, prefix + col] = probs[col].values
        out.loc[exp_log_vol.index, prefix + "exp_log_vol"] = exp_log_vol.values

        return out

    # -------------------------
    # Save / load
    # -------------------------
    def save(self, path: str):
        """
        Save model, scaler, and config to a single joblib file.
        """
        self._check_fitted()
        payload: Dict[str, Any] = {
            "config": self.config,
            "model": self.model,
            "scaler": self.scaler,
            "state_order_": self.state_order_,
            "fitted_": self.fitted_,
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: str) -> "VolatilityHMM":
        payload = joblib.load(path)
        config = payload["config"]
        obj = cls(config=config)
        obj.model = payload["model"]
        obj.scaler = payload["scaler"]
        obj.state_order_ = payload["state_order_"]
        obj.fitted_ = payload["fitted_"]
        return obj


# =========================
# Macro HMM (FRED) builder
# =========================
def fetch_fred_series(series_id: str, start: str, end: str, api_key: str) -> pd.Series:
    """
    Fetch a monthly FRED series and return a Series indexed by month-end.
    """
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start,
        "observation_end": end,
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()["observations"]
    df = pd.DataFrame(data)
    df["value"] = pd.to_numeric(df["value"].replace(".", np.nan), errors="coerce")
    df["date"] = pd.to_datetime(df["date"]) + pd.offsets.MonthEnd(0)
    return df.set_index("date")["value"].rename(series_id)


def build_macro_hmm(
    start_date: str = "1970-01-01",
    end_date: str = "2023-01-01",
    n_states: int = 3,
    random_state: int = 42,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Fetch macro data from FRED, engineer spreads/growth, fit a Gaussian HMM,
    and save regimes plus model artifacts.
    """
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise RuntimeError("FRED_API_KEY not set in environment.")

    series_ids = [
        "GS10",      # 10Y Treasury
        "TB3MS",     # 3M T-Bill
        "BAA",       # Baa Corporate
        "AAA",       # Aaa Corporate
        "INDPRO",    # Industrial Production
        "CPIAUCSL",  # CPI
        "UNRATE",    # Unemployment
        "USRECM",    # Recession indicator (for reference)
    ]

    print("Fetching FRED series...")
    df_list = [fetch_fred_series(s, start_date, end_date, api_key) for s in series_ids]
    df = pd.concat(df_list, axis=1).dropna(how="all").sort_index()

    # Feature engineering (monthly)
    df["TERM_SPREAD"] = df["GS10"] - df["TB3MS"]
    df["CREDIT_SPREAD"] = df["BAA"] - df["AAA"]
    df["INDPRO_YoY"] = pd.Series(np.log(df["INDPRO"])).diff(12) * 100
    df["CPI_YoY"] = pd.Series(np.log(df["CPIAUCSL"])).diff(12) * 100
    df["UNRATE_Delta"] = df["UNRATE"].diff(6)

    features = ["TERM_SPREAD", "CREDIT_SPREAD", "INDPRO_YoY", "CPI_YoY", "UNRATE_Delta"]
    df_model = df[features].dropna()

    X = df_model.values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Training macro HMM...")
    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=1000,
        random_state=random_state,
    )
    model.fit(X_scaled)

    # Order states by CREDIT_SPREAD (low -> high risk)
    credit_idx = features.index("CREDIT_SPREAD")
    state_means = model.means_[:, credit_idx]
    order = np.argsort(state_means)
    mapping = {raw: ordered for ordered, raw in enumerate(order)}

    hidden_raw = model.predict(X_scaled)
    hidden_sorted = np.array([mapping[s] for s in hidden_raw])
    probs_raw = model.predict_proba(X_scaled)
    probs_sorted = probs_raw[:, order]

    out = df_model.copy()
    out["hmm_state"] = hidden_sorted
    for k in range(n_states):
        out[f"prob_state_{k}"] = probs_sorted[:, k]

    # Save artifacts
    project_root = Path(__file__).resolve().parents[1]
    out_dir = output_dir or project_root / "data" / "master"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "hmm_macro_regimes_monthly.csv"
    model_path = out_dir / "hmm_model_trained.pkl"
    scaler_path = out_dir / "hmm_scaler.pkl"

    out.to_csv(csv_path)
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    print(f"Saved regimes to {csv_path}")
    print(f"Saved model to   {model_path}")
    print(f"Saved scaler to  {scaler_path}")

    return {
        "data": out,
        "model": model,
        "scaler": scaler,
        "state_order": order,
    }


if __name__ == "__main__":
    # Build macro HMM using FRED data and save outputs
    build_macro_hmm()
