import pandas as pd
import numpy as np

def compute_order_imbalance(df, time_col="timestamp", tenor_col="tenor", side_col="side"):
    """
    Compute bid-ask imbalance in the last hour, grouped by tenor buckets.
    
    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns [time_col, tenor_col, side_col].
        side_col should be "BID" or "ASK".
        tenor_col should be numeric (years) or convertible to numeric.
    time_col : str
        Name of the timestamp column (must be datetime type).
    tenor_col : str
        Name of the tenor column (in years).
    side_col : str
        Column indicating whether order is BID or ASK.
    
    Returns
    -------
    pd.DataFrame
        Tenor-bucket-level imbalance for the last hour.
    """

    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])

    # Ensure tenor is numeric (assuming input like '5Y' or int)
    if df[tenor_col].dtype == object:
        df[tenor_col] = df[tenor_col].str.replace("Y", "").astype(float)

    # Define tenor buckets
    bins = [0, 2, 5, 9, 15, 30, np.inf]
    labels = ["0-2", "2-5", "5-9", "10-15", "16-30", "30+"]
    df["tenor_bucket"] = pd.cut(df[tenor_col], bins=bins, labels=labels, right=True)

    # Filter last hour
    cutoff = df[time_col].max() - pd.Timedelta(hours=1)
    df_last_hour = df[df[time_col] >= cutoff]

    # Count bids/asks per tenor bucket
    counts = df_last_hour.groupby(["tenor_bucket", side_col]).size().unstack(fill_value=0)

    # Compute imbalance
    counts["imbalance"] = (counts.get("BID", 0) - counts.get("ASK", 0)) / (
        counts.get("BID", 0) + counts.get("ASK", 0) + 1e-9
    )

    return counts.reset_index()