from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

from .config import ProcessingConfig


def butter_lowpass(cutoff_hz: float, fs_hz: float, order: int = 3):
    nyq = 0.5 * fs_hz
    normal_cutoff = float(cutoff_hz) / nyq
    return butter(order, normal_cutoff, btype="low", analog=False)


def butter_lowpass_filter(x: np.ndarray, cutoff_hz: float, fs_hz: float, order: int = 3) -> np.ndarray:
    """Zero-phase low-pass filter. Expects finite array; caller fills NaNs."""
    b, a = butter_lowpass(cutoff_hz, fs_hz, order)
    return filtfilt(b, a, x)


def detect_rule_key(filename_lower: str) -> str | None:
    """Return '2s', '6s', or None based on substring match."""
    name = filename_lower.lower()
    if "2s" in name:
        return "2s"
    if "6s" in name:
        return "6s"
    return None


def _normalize_flag_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().fillna("")


def _dilate_bool_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    """Expand True region by +/- radius samples."""
    radius = int(radius) if radius is not None else 0
    if radius <= 0:
        return mask.copy()

    n = len(mask)
    out = mask.copy()
    idx = np.flatnonzero(mask)
    for i in idx:
        a = max(0, i - radius)
        b = min(n, i + radius + 1)
        out[a:b] = True
    return out


def mark_invalid_pupil_samples(df: pd.DataFrame, config: ProcessingConfig) -> pd.DataFrame:
    """
    Set pupil samples to NaN for blink/low-accuracy flags, plus neighbor expansion.

    This function can handle both eyes, but run_pipeline() drops right-eye columns first,
    so in the verified app workflow it effectively touches LEFT EYE ONLY.
    """
    out = df.copy()

    for pupil_col, eye_state_col, err_col in [
        (config.pupil_left_col, config.eye_state_left, config.err_left_col),
        (config.pupil_right_col, config.eye_state_right, config.err_right_col),
    ]:
        if pupil_col not in out.columns:
            continue

        out[pupil_col] = pd.to_numeric(out[pupil_col], errors="coerce")
        base_mask = np.zeros(len(out), dtype=bool)

        if eye_state_col in out.columns:
            state = _normalize_flag_series(out[eye_state_col])
            base_mask |= (state == "closed").to_numpy()

        if err_col in out.columns:
            err = _normalize_flag_series(out[err_col])
            base_mask |= err.isin(["data_lowaccuracy", "data_unreliable"]).to_numpy()

        expanded = _dilate_bool_mask(base_mask, config.neighbor_radius)
        out.loc[expanded, pupil_col] = np.nan

    return out


def apply_delta_exclusion_preinterp(df: pd.DataFrame, pupil_col: str, delta_thresh: float) -> pd.DataFrame:
    """Set sample to NaN when abs(diff) exceeds threshold, before interpolation."""
    out = df.copy()
    if pupil_col not in out.columns:
        return out

    x = pd.to_numeric(out[pupil_col], errors="coerce")
    jump = x.diff().abs() > float(delta_thresh)
    out.loc[jump.fillna(False), pupil_col] = np.nan
    return out


def pre_interp_exclusion_rate(df: pd.DataFrame, pupil_col: str) -> float:
    if pupil_col not in df.columns or len(df) == 0:
        return float("nan")
    s = pd.to_numeric(df[pupil_col], errors="coerce")
    return float(s.isna().mean())


def trim_time_window(df: pd.DataFrame, time_col: str, trim_start_s: float, trim_end_s: float) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    t = pd.to_numeric(df[time_col], errors="coerce")
    df2 = df.copy()
    df2[time_col] = t
    df2 = df2.dropna(subset=[time_col]).copy()
    if df2.empty:
        return df2

    tmin = float(df2[time_col].iloc[0])
    tmax = float(df2[time_col].iloc[-1])
    start_t = tmin + float(trim_start_s)
    end_t = tmax - float(trim_end_s)

    if end_t <= start_t:
        return df2.iloc[0:0].copy()

    return df2[(df2[time_col] >= start_t) & (df2[time_col] <= end_t)].copy()


def blank_bad_cycles_by_ratio(
    df: pd.DataFrame,
    *,
    time_col: str,
    cycle_start: float,
    cycle_len: float,
    quality_col: str,
    cols_to_blank: list[str],
    bad_ratio: float,
) -> tuple[pd.DataFrame, dict]:
    """If NaN ratio within a cycle >= bad_ratio, blank that entire cycle."""
    out = df.copy()
    if out.empty:
        return out, {"n_cycles": 0, "n_bad_cycles": 0, "rows_blank_ratio": 0.0}

    out = out[out[time_col] >= float(cycle_start)].copy()
    if out.empty:
        return out, {"n_cycles": 0, "n_bad_cycles": 0, "rows_blank_ratio": 0.0}

    cid = np.floor((out[time_col] - float(cycle_start)) / float(cycle_len)).astype(int)
    out["_cycle_id"] = cid

    if quality_col not in out.columns:
        out.drop(columns=["_cycle_id"], inplace=True)
        return out, {
            "n_cycles": int(cid.nunique()),
            "n_bad_cycles": 0,
            "rows_blank_ratio": 0.0,
            "bad_cycles": [],
            "cycle_start": float(cycle_start),
            "cycle_len": float(cycle_len),
        }

    q = pd.to_numeric(out[quality_col], errors="coerce")
    nan_mask = q.isna()

    grp = out.groupby("_cycle_id")
    nan_counts = grp.apply(lambda g: nan_mask.loc[g.index].sum())
    total_counts = grp.size()
    nan_ratio = nan_counts / total_counts

    bad_cycles = set(nan_ratio[nan_ratio >= float(bad_ratio)].index.tolist())
    bad_row_mask = out["_cycle_id"].isin(bad_cycles)

    for col in cols_to_blank:
        if col in out.columns:
            out.loc[bad_row_mask, col] = np.nan

    rows_blank_ratio = float(bad_row_mask.sum()) / float(len(out)) if len(out) else 0.0

    out.drop(columns=["_cycle_id"], inplace=True)
    return out, {
        "n_cycles": int(total_counts.shape[0]),
        "n_bad_cycles": int(len(bad_cycles)),
        "rows_blank_ratio": rows_blank_ratio,
        "bad_cycles": sorted(list(bad_cycles)),
        "cycle_start": float(cycle_start),
        "cycle_len": float(cycle_len),
    }


def run_pipeline(
    csv_path: Path,
    output_dir: Path,
    config: ProcessingConfig,
) -> tuple[pd.DataFrame, dict, Path | None]:
    """
    Verified app version of integrated_filter_fin_keep_blanked_cycles_nan.ipynb.

    This is LEFT-EYE ONLY and restores NaNs for cycle-blanked segments after
    interpolation/resampling/filtering so blanked cycles remain blank.
    """
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)

    df = pd.read_csv(csv_path, low_memory=False)

    if config.time_col not in df.columns:
        raise ValueError(f"Missing required time column: {config.time_col}")

    df[config.time_col] = pd.to_numeric(df[config.time_col], errors="coerce")
    df = df.dropna(subset=[config.time_col]).copy()
    if df.empty:
        raise ValueError(f"No valid time samples in: {csv_path.name}")

    # Sort by time and normalize to start at 0
    df = df.sort_values(config.time_col).reset_index(drop=True)
    df[config.time_col] -= float(df[config.time_col].iloc[0])

    # LEFT-EYE ONLY: drop right-eye related columns before QC
    df = df.drop(
        columns=[config.pupil_right_col, config.eye_state_right, config.err_right_col],
        errors="ignore",
    )

    # 1) Mark invalid left samples -> NaN
    df = mark_invalid_pupil_samples(df, config)

    # 2) Delta exclusion before interpolation, left only
    if config.pupil_left_col in df.columns:
        df = apply_delta_exclusion_preinterp(
            df,
            config.pupil_left_col,
            config.delta_threshold,
        )

    # 3) Rule-based trim + cycle blanking, left-eye QC
    rule_key = detect_rule_key(csv_path.name.lower())
    cycle_info = {"n_cycles": 0, "n_bad_cycles": 0, "rows_blank_ratio": 0.0}
    cycle_blanked_rows_percent = 0.0
    bad_cycles: list[int] = []
    cycle_start_s = 0.0
    cycle_len_s = 0.0

    if rule_key is not None:
        rule = config.rules[rule_key]
        df = trim_time_window(
            df,
            config.time_col,
            rule["trim_start"],
            rule["trim_end"],
        )

        if config.pupil_left_col in df.columns:
            df, cycle_info = blank_bad_cycles_by_ratio(
                df,
                time_col=config.time_col,
                cycle_start=rule["cycle_start"],
                cycle_len=rule["cycle_len"],
                quality_col=config.pupil_left_col,
                cols_to_blank=[config.pupil_left_col],
                bad_ratio=rule["bad_ratio"],
            )
            cycle_blanked_rows_percent = float(cycle_info.get("rows_blank_ratio", 0.0)) * 100.0
            bad_cycles = [int(x) for x in (cycle_info.get("bad_cycles", []) or [])]
            cycle_start_s = float(cycle_info.get("cycle_start", rule["cycle_start"]))
            cycle_len_s = float(cycle_info.get("cycle_len", rule["cycle_len"]))

    # 4) Pre-interpolation exclusion rate, left only
    pre_left = (
        pre_interp_exclusion_rate(df, config.pupil_left_col)
        if config.pupil_left_col in df.columns
        else float("nan")
    )

    # 4.5) Skip if too many bad cycles
    skip_threshold = config.max_bad_cycles(rule_key)
    skipped_due_to_bad_cycles = False
    if skip_threshold is not None:
        if int(cycle_info.get("n_bad_cycles", 0)) >= int(skip_threshold):
            skipped_due_to_bad_cycles = True

    if skipped_due_to_bad_cycles:
        summary = {
            "file": csv_path.name,
            "rule": rule_key,
            "pre_interp_excl_left": pre_left,
            "cycle_blanked_rows_percent": cycle_blanked_rows_percent,
            "n_cycles": int(cycle_info.get("n_cycles", 0)),
            "n_bad_cycles": int(cycle_info.get("n_bad_cycles", 0)),
            "skip_threshold_bad_cycles": skip_threshold,
            "skipped": True,
            "processed_output": None,
            "error": "Skipped because n_bad_cycles reached threshold.",
        }
        return pd.DataFrame(), summary, None

    # 5) PCHIP interpolation on raw timeline, left only
    # Then restore blanked-cycle NaNs so bad cycles are not filled.
    if config.pupil_left_col in df.columns:
        df[config.pupil_left_col] = pd.to_numeric(
            df[config.pupil_left_col],
            errors="coerce",
        ).interpolate("pchip")

        if bad_cycles:
            t_df = pd.to_numeric(df[config.time_col], errors="coerce").to_numpy()
            bad_mask_df = np.zeros(len(df), dtype=bool)
            for cid in bad_cycles:
                a = cycle_start_s + cid * cycle_len_s
                b = a + cycle_len_s
                bad_mask_df |= (t_df >= a) & (t_df < b)
            df.loc[bad_mask_df, config.pupil_left_col] = np.nan

    # 6) Resample to FS
    t0 = df[config.time_col].to_numpy()
    t = np.arange(float(t0[0]), float(t0[-1]), 1.0 / float(config.fs))
    out = pd.DataFrame({config.time_col: t})

    if config.pupil_left_col in df.columns:
        out[config.pupil_left_col] = np.interp(
            t,
            t0,
            df[config.pupil_left_col].to_numpy(),
        )

    # Build resampled-time mask for blanked cycles
    if bad_cycles:
        bad_mask_t = np.zeros(len(t), dtype=bool)
        for cid in bad_cycles:
            a = cycle_start_s + cid * cycle_len_s
            b = a + cycle_len_s
            bad_mask_t |= (t >= a) & (t < b)
    else:
        bad_mask_t = None

    if bad_mask_t is not None and config.pupil_left_col in out.columns:
        out.loc[bad_mask_t, config.pupil_left_col] = np.nan

    # 7) Low-pass filter, then restore blanked-cycle NaNs
    if config.pupil_left_col in out.columns:
        x = (
            pd.to_numeric(out[config.pupil_left_col], errors="coerce")
            .interpolate()
            .bfill()
            .ffill()
            .to_numpy()
        )
        out[config.filtered_left_col] = butter_lowpass_filter(
            x,
            config.lp_cutoff,
            config.fs,
            config.lp_order,
        )
        if bad_mask_t is not None:
            out.loc[bad_mask_t, config.filtered_left_col] = np.nan

    # 8) Output: exactly the 3-column left-eye processed file
    output_dir.mkdir(exist_ok=True, parents=True)
    out_path = output_dir / f"processed_{csv_path.stem}.csv"
    out.to_csv(out_path, index=False)

    summary = {
        "file": csv_path.name,
        "rule": rule_key or "none",
        "pre_interp_excl_left": pre_left,
        "cycle_blanked_rows_percent": cycle_blanked_rows_percent,
        "n_cycles": int(cycle_info.get("n_cycles", 0)),
        "n_bad_cycles": int(cycle_info.get("n_bad_cycles", 0)),
        "skip_threshold_bad_cycles": skip_threshold,
        "skipped": False,
        "processed_output": str(out_path),
        "error": "",
    }

    return out, summary, out_path
