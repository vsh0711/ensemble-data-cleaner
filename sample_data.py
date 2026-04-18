import numpy as np
import pandas as pd


def generate_sample_data(n=500, anomaly_fraction=0.05, random_state=42):
    """
    Generates a synthetic BFSI-style dataset with injected anomalies.
    Simulates the kind of banking customer data Vishalini worked with at TCS.
    No real customer data — entirely synthetic.
    """
    rng = np.random.default_rng(random_state)
    n_anomaly = int(n * anomaly_fraction)
    n_clean   = n - n_anomaly

    # Clean records — realistic banking distributions
    clean = pd.DataFrame({
        'customer_id':    range(1, n_clean + 1),
        'age':            rng.integers(22, 65, n_clean),
        'credit_score':   rng.normal(680, 80, n_clean).clip(300, 850).round(),
        'account_balance': rng.exponential(50000, n_clean).round(2),
        'monthly_txn_count': rng.integers(1, 40, n_clean),
        'credit_limit':   rng.choice([50000,100000,200000,300000,500000], n_clean),
        'loan_amount':    rng.exponential(200000, n_clean).round(2),
        'risk_rating':    rng.integers(1, 6, n_clean),
        'months_active':  rng.integers(1, 120, n_clean),
        'avg_txn_value':  rng.normal(3000, 1500, n_clean).clip(100, 20000).round(2),
        'is_anomaly_injected': 0
    })

    # Anomalous records — corrupted / extreme values
    anomalies = pd.DataFrame({
        'customer_id':    range(n_clean + 1, n + 1),
        'age':            rng.choice([5, 150, 200, -1, 999], n_anomaly),
        'credit_score':   rng.choice([1, 999, 1500, -500], n_anomaly),
        'account_balance': rng.choice([-999999, 9999999, 0.001], n_anomaly),
        'monthly_txn_count': rng.integers(500, 5000, n_anomaly),
        'credit_limit':   rng.choice([-50000, 99999999], n_anomaly),
        'loan_amount':    rng.choice([-1, 999999999], n_anomaly),
        'risk_rating':    rng.choice([0, 99, -5], n_anomaly),
        'months_active':  rng.choice([-10, 9999], n_anomaly),
        'avg_txn_value':  rng.choice([-5000, 500000], n_anomaly),
        'is_anomaly_injected': 1
    })

    df = pd.concat([clean, anomalies], ignore_index=True)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    df = df.drop('customer_id', axis=1)
    return df