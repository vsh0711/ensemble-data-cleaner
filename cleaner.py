import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler


class EnsembleDataCleaner:


    def __init__(self, contamination=0.05, zscore_threshold=3.0, random_state=42):
        self.contamination      = contamination
        self.zscore_threshold   = zscore_threshold
        self.random_state       = random_state
        self.scaler             = StandardScaler()
        self.iso_forest         = IsolationForest(
            contamination=contamination,
            n_estimators=150,
            random_state=random_state
        )
        self.lof = LocalOutlierFactor(
            n_neighbors=20,
            contamination=contamination
        )
        self.is_fitted          = False
        self.numeric_cols_      = None
        self.results_           = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_detect(self, df: pd.DataFrame) -> pd.DataFrame:
        
        self.numeric_cols_ = df.select_dtypes(include=[np.number]).columns.tolist()

        if len(self.numeric_cols_) == 0:
            raise ValueError("No numeric columns found. Cannot run anomaly detection.")

        X = df[self.numeric_cols_].copy()
        X = X.fillna(X.median())
        X_scaled = self.scaler.fit_transform(X)

        # Method 1 — Isolation Forest
        iso_labels  = self.iso_forest.fit_predict(X_scaled)
        iso_scores  = -self.iso_forest.score_samples(X_scaled)   # higher = worse
        iso_flags   = (iso_labels == -1).astype(int)

        # Method 2 — Z-score (per column, take row maximum)
        z_matrix    = np.abs(X_scaled)
        z_max       = z_matrix.max(axis=1)
        zscore_flags = (z_max > self.zscore_threshold).astype(int)

        # Method 3 — Local Outlier Factor
        lof_labels  = self.lof.fit_predict(X_scaled)
        lof_flags   = (lof_labels == -1).astype(int)

        # Ensemble consensus — flag if 2+ methods agree
        votes       = iso_flags + zscore_flags + lof_flags
        ensemble    = (votes >= 2).astype(int)

        result = df.copy()
        result['iso_flag']      = iso_flags
        result['zscore_flag']   = zscore_flags
        result['lof_flag']      = lof_flags
        result['anomaly_score'] = np.round(iso_scores, 4)
        result['anomaly']       = ensemble

        self.is_fitted  = True
        self.results_   = result
        return result

    def get_clean(self) -> pd.DataFrame:
        """Return only the clean rows (anomaly == 0), original columns only."""
        self._check_fitted()
        original_cols = [c for c in self.results_.columns
                         if c not in ['iso_flag','zscore_flag',
                                      'lof_flag','anomaly_score','anomaly']]
        return self.results_[self.results_['anomaly'] == 0][original_cols].reset_index(drop=True)

    def get_anomalies(self) -> pd.DataFrame:
        """Return only the anomalous rows with all flag columns."""
        self._check_fitted()
        return self.results_[self.results_['anomaly'] == 1].reset_index(drop=True)

    def summary(self) -> dict:
        """Return a summary dictionary for display in the UI."""
        self._check_fitted()
        total       = len(self.results_)
        n_anomaly   = int(self.results_['anomaly'].sum())
        n_clean     = total - n_anomaly

        method_counts = {
            'Isolation Forest':      int(self.results_['iso_flag'].sum()),
            'Z-score':               int(self.results_['zscore_flag'].sum()),
            'Local Outlier Factor':  int(self.results_['lof_flag'].sum()),
        }

        # Which columns contribute most to anomalies
        anomaly_rows = self.results_[self.results_['anomaly'] == 1]
        clean_rows   = self.results_[self.results_['anomaly'] == 0]

        col_drift = {}
        for col in self.numeric_cols_:
            if col in anomaly_rows.columns:
                a_mean = anomaly_rows[col].mean()
                c_mean = clean_rows[col].mean()
                if c_mean != 0:
                    col_drift[col] = abs((a_mean - c_mean) / (abs(c_mean) + 1e-9))

        top_cols = sorted(col_drift, key=col_drift.get, reverse=True)[:5]

        return {
            'total':         total,
            'n_anomaly':     n_anomaly,
            'n_clean':       n_clean,
            'anomaly_pct':   round(n_anomaly / total * 100, 2),
            'method_counts': method_counts,
            'top_drift_cols': top_cols,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError("Call fit_detect() before accessing results.")