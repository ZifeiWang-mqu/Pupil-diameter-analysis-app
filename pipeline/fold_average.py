import numpy as np
import pandas as pd

from .config import ProcessingConfig


def block_fold_average_exact(
    df: pd.DataFrame,
    *,
    block_len_s: float,
    block_start_s: float,
    time_min: float,
    time_max_exclusive: float,
    time_col: str | None = None,
) -> pd.DataFrame:
    """Exact app version of block2s/block6s average notebooks."""
    if df is None or df.empty:
        raise ValueError("Input DataFrame is empty.")

    # Same as the notebook after reading CSV: blank separator columns become Unnamed columns
    # and are dropped. Because the app passes the merged DataFrame in memory,
    # we must also drop columns whose name is exactly empty string.
    col_as_str = df.columns.astype(str)
    keep = (~col_as_str.str.match(r"^Unnamed")) & (col_as_str.str.strip() != "")
    df = df.loc[:, keep].copy()

    time_col = time_col or df.columns[0]
    t = pd.to_numeric(df[time_col], errors="coerce")

    valid = t.notna()
    df = df.loc[valid].copy()
    t = t.loc[valid].reset_index(drop=True)
    df = df.reset_index(drop=True)

    dt = float(np.median(np.diff(t.to_numpy())))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Could not estimate dt from time column.")

    points_per_block = int(round(float(block_len_s) / dt))
    if points_per_block <= 1:
        raise ValueError(f"Invalid points_per_block: {points_per_block} (dt={dt})")

    mask = (t >= float(time_min)) & (t < float(time_max_exclusive))
    df_use = df.loc[mask].copy()
    t_use = t.loc[mask].to_numpy()

    offset = (t_use - float(block_start_s)) % float(block_len_s)
    sample_idx = np.rint(offset / dt).astype(int)

    ok = (sample_idx >= 0) & (sample_idx < points_per_block)
    df_use = df_use.loc[ok].copy()
    sample_idx = sample_idx[ok]

    df_use["_sample_idx"] = sample_idx

    data_cols = [c for c in df_use.columns if c not in [time_col, "_sample_idx"]]
    df_num = df_use[data_cols].apply(pd.to_numeric, errors="coerce")
    df_num["_sample_idx"] = df_use["_sample_idx"].to_numpy()

    mean_by_idx = df_num.groupby("_sample_idx", sort=True).mean(numeric_only=True)
    out_time = float(block_start_s) + mean_by_idx.index.to_numpy() * dt

    out = pd.DataFrame({time_col: out_time})
    out = pd.concat([out.reset_index(drop=True), mean_by_idx.reset_index(drop=True)], axis=1)
    return out


def block2s_average_from_merged(df: pd.DataFrame, config: ProcessingConfig) -> pd.DataFrame:
    rule = config.rules["2s"]
    return block_fold_average_exact(
        df,
        block_len_s=2.0,
        block_start_s=rule["cycle_start"],
        time_min=rule["fold_time_min"],
        time_max_exclusive=rule["fold_time_max_exclusive"],
        time_col=None,
    )


def block6s_average_from_merged(df: pd.DataFrame, config: ProcessingConfig) -> pd.DataFrame:
    rule = config.rules["6s"]
    return block_fold_average_exact(
        df,
        block_len_s=6.0,
        block_start_s=rule["cycle_start"],
        time_min=rule["fold_time_min"],
        time_max_exclusive=rule["fold_time_max_exclusive"],
        time_col=None,
    )
