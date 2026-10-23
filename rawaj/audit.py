
"""Data audit for Rawaj Pay transfer extracts."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def is_missing(series):
    """Treat NaN, None, and blank strings as missing."""
    blanks = series.map(
        lambda x: isinstance(x, str) and x.strip() == ""
    )
    return series.isna() | blanks


def audit(
    df: pd.DataFrame,
    *,
    target: str,
    id_col: str,
    time_col: str,
) -> dict:
    """Audit a transfer extract before model training."""

    # 1. Check required columns
    for col in (target, id_col, time_col):
        if col not in df.columns:
            raise KeyError(col)

    n_rows = len(df)

    # 2. Missing values
    missing_masks = {
        col: is_missing(df[col])
        for col in df.columns
    }

    missing = {
        col: float(mask.mean()) if n_rows else 0.0
        for col, mask in missing_masks.items()
    }

    # 3. Duplicate IDs
    ids = df.loc[~missing_masks[id_col], id_col]
    duplicate_ids = sorted(
        ids[ids.duplicated(keep=False)].unique().tolist()
    )

    # 4. Exact duplicate rows
    duplicate_rows = int(df.duplicated().sum())

    # 5. Constant columns, excluding target
    constant_columns = sorted([
        col for col in df.columns
        if col != target
        and df.loc[~missing_masks[col], col].nunique() <= 1
    ])

    # 6. Unlabelled rows and positive rate
    target_missing = missing_masks[target]
    unlabelled_rows = int(target_missing.sum())

    labelled = df.loc[~target_missing, target]
    positive_rate = (
        float((labelled == 1).mean())
        if len(labelled) else 0.0
    )

    # 7. Timestamp validation
    time_mask = ~missing_masks[time_col]
    timestamps = pd.to_datetime(
        df.loc[time_mask, time_col],
        errors="coerce",
    )

    bad_timestamps = int(timestamps.isna().sum())
    valid_timestamps = timestamps.dropna()

    if len(valid_timestamps):
        time_range = [
            valid_timestamps.min().isoformat(),
            valid_timestamps.max().isoformat(),
        ]
    else:
        time_range = [None, None]

    # 8. Potential target leakage
    leakage_suspects = []

    labelled_positions = np.flatnonzero(
        (~target_missing).to_numpy()
    )
    y = df[target].iloc[labelled_positions].to_numpy()

    if len(y) >= 20 and set(np.unique(y)) == {0, 1}:
        for col in df.columns:
            if col in (target, id_col, time_col):
                continue

            # A column with only one distinct value,
            # counting missingness as a value, is not a suspect.
            col_missing = missing_masks[col].iloc[
                labelled_positions
            ].to_numpy()

            values = df[col].iloc[labelled_positions]
            distinct_count = values.nunique(dropna=False)

            if distinct_count <= 1:
                continue

            suspect = False

            # Check numeric/boolean values against the target.
            nonmissing_positions = np.flatnonzero(~col_missing)
            observed = values.iloc[nonmissing_positions]
            observed_y = y[nonmissing_positions]

            if (
                len(observed) >= 20
                and observed.nunique() > 1
                and (
                    pd.api.types.is_numeric_dtype(df[col])
                    or pd.api.types.is_bool_dtype(df[col])
                )
                and len(np.unique(observed_y)) == 2
            ):
                auc = roc_auc_score(
                    observed_y,
                    observed.astype(float).to_numpy(),
                )

                if auc >= 0.98 or auc <= 0.02:
                    suspect = True

            # Check whether missingness itself predicts the target.
            if not suspect and len(np.unique(col_missing)) == 2:
                auc_missing = roc_auc_score(
                    y,
                    col_missing.astype(int),
                )

                if auc_missing >= 0.98 or auc_missing <= 0.02:
                    suspect = True

            if suspect:
                leakage_suspects.append(col)

    # 9. Return exactly the required keys
    return {
        "n_rows": int(n_rows),
        "missing": missing,
        "duplicate_ids": duplicate_ids,
        "duplicate_rows": duplicate_rows,
        "constant_columns": constant_columns,
        "unlabelled_rows": unlabelled_rows,
        "positive_rate": positive_rate,
        "bad_timestamps": bad_timestamps,
        "time_range": time_range,
        "leakage_suspects": sorted(leakage_suspects),
    }
