import re
from io import BytesIO

import numpy as np
import pandas as pd


COL_PATTERN = re.compile(
    r"sub(?P<sub>\d+).*?stimul(?:us|i)(?P<stim>\d+)",
    re.IGNORECASE,
)


def parse_subject_stimulus(col: str) -> tuple[str, str]:
    """Extract subject and stimulus labels from a block-average column name."""
    m = COL_PATTERN.search(str(col))
    if not m:
        raise ValueError(
            f"列名 '{col}' から subject/stimulus を抽出できません。"
            " 例: sub15_2s_stimuli1_left のような形式を想定しています。"
        )

    subject = f"sub{int(m.group('sub')):02d}"
    stimulus = f"stimulus{int(m.group('stim'))}"
    return subject, stimulus


def _sort_stimulus_labels(labels):
    return sorted(labels, key=lambda s: int(re.search(r"\d+", str(s)).group()))


def wide_to_long(values_by_col: dict[str, float]) -> pd.DataFrame:
    """Convert one DV's wide values into long format for pingouin.rm_anova."""
    rows = []
    for col, val in values_by_col.items():
        subject, stimulus = parse_subject_stimulus(col)
        rows.append({"subject": subject, "stimulus": stimulus, "value": val})

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("ANOVA用のlong DataFrameが空です。列名に stimuli1〜5 が含まれているか確認してください。")

    out["stimulus"] = pd.Categorical(
        out["stimulus"],
        ordered=True,
        categories=_sort_stimulus_labels(out["stimulus"].unique()),
    )
    return out.sort_values(["subject", "stimulus"]).reset_index(drop=True)


def run_rm_anova(long_df: pd.DataFrame) -> pd.DataFrame:
    """Run one-way repeated-measures ANOVA with stimulus as within factor."""
    try:
        import pingouin as pg
    except ImportError as exc:
        raise ImportError(
            "ANOVA機能には pingouin が必要です。ターミナルで `pip install pingouin openpyxl` を実行してください。"
        ) from exc

    return pg.rm_anova(
        data=long_df,
        dv="value",
        within="stimulus",
        subject="subject",
        detailed=True,
        effsize="ng2",
    )


def pick_exact_time_row(
    df: pd.DataFrame,
    time_col: str,
    target_t: float,
    tol: float = 1e-9,
) -> tuple[float, int, pd.Series]:
    """Pick a row by exact relative time, matching the uploaded rmANOVA notebook."""
    t = pd.to_numeric(df[time_col], errors="coerce").to_numpy()
    mask = np.isclose(t, float(target_t), atol=float(tol), rtol=0.0)
    idxs = np.where(mask)[0]

    if len(idxs) == 0:
        nearest_idx = int(np.nanargmin(np.abs(t - float(target_t))))
        raise ValueError(
            f"時刻 {target_t} が time_col='{time_col}' に見つかりません。"
            f" 近傍: idx={nearest_idx}, t={t[nearest_idx]}"
        )

    idx = int(idxs[0])
    return float(t[idx]), idx, df.iloc[idx]


def _target_label_from_time(target_t: float) -> str:
    text = str(target_t).replace(".", "p")
    return f"t{text}"


def run_rm_anova_from_block_average(
    block_df: pd.DataFrame,
    *,
    target_times_rel: dict[str, float],
    filter_suffix: str | None = "_left",
    exact_time_tolerance: float = 1e-9,
) -> dict[str, pd.DataFrame]:
    """
    Function version of rmANOVA_2s_27thFeb.ipynb.

    Input is block2s_fold_mean.csv or block6s_fold_mean.csv as a DataFrame.
    It computes ANOVA for:
    - mean
    - specified relative timepoints
    - max
    - min
    - range = max - min
    """
    if block_df is None or block_df.empty:
        raise ValueError("ANOVA入力のblock-average DataFrameが空です。")

    df = block_df.copy()

    # Drop separator columns just in case a merged-style dataframe is passed accidentally.
    col_as_str = df.columns.astype(str)
    keep = (~col_as_str.str.match(r"^Unnamed")) & (col_as_str.str.strip() != "")
    df = df.loc[:, keep].copy()

    time_col_abs = df.columns[0]
    df[time_col_abs] = pd.to_numeric(df[time_col_abs], errors="coerce")

    rel_time_col = "__t_rel"
    df[rel_time_col] = df[time_col_abs] - df[time_col_abs].iloc[0]

    data_cols = [c for c in df.columns if c not in (time_col_abs, rel_time_col)]

    if filter_suffix is not None:
        data_cols = [
            c for c in data_cols
            if str(c).lower().endswith(str(filter_suffix).lower())
        ]

    if not data_cols:
        raise ValueError(
            "ANOVA対象列がありません。block fold meanの列名が `subXX_..._stimuliN_left` 形式か確認してください。"
        )

    # (A) mean DV
    mean_values = {}
    for c in data_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        mean_values[c] = float(np.nanmean(s.to_numpy()))

    long_mean = wide_to_long(mean_values)
    aov_mean = run_rm_anova(long_mean)

    # (B) timepoint DVs
    timepoint_meta = []
    timepoint_longs: dict[str, pd.DataFrame] = {}
    timepoint_aovs: dict[str, pd.DataFrame] = {}

    for label, target_t_rel in target_times_rel.items():
        used_t_rel, used_idx, row = pick_exact_time_row(
            df,
            rel_time_col,
            target_t_rel,
            tol=exact_time_tolerance,
        )
        used_t_abs = float(df.loc[used_idx, time_col_abs])

        values = {}
        for c in data_cols:
            values[c] = float(pd.to_numeric(row[c], errors="coerce"))

        long_tp = wide_to_long(values)
        aov_tp = run_rm_anova(long_tp)

        timepoint_longs[label] = long_tp
        timepoint_aovs[label] = aov_tp
        timepoint_meta.append(
            {
                "label": label,
                "target_t_rel": target_t_rel,
                "used_t_rel": used_t_rel,
                "used_t_abs": used_t_abs,
                "used_idx": used_idx,
            }
        )

    meta_df = pd.DataFrame(timepoint_meta)

    # (C) max / min / range DVs
    max_values = {}
    min_values = {}
    range_values = {}

    for c in data_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().any():
            v_max = float(s.max(skipna=True))
            v_min = float(s.min(skipna=True))
            v_range = v_max - v_min
        else:
            v_max = np.nan
            v_min = np.nan
            v_range = np.nan

        max_values[c] = v_max
        min_values[c] = v_min
        range_values[c] = v_range

    long_max = wide_to_long(max_values)
    long_min = wide_to_long(min_values)
    long_range = wide_to_long(range_values)

    aov_max = run_rm_anova(long_max)
    aov_min = run_rm_anova(long_min)
    aov_range = run_rm_anova(long_range)

    sheets: dict[str, pd.DataFrame] = {
        "ANOVA_mean": aov_mean,
        "ANOVA_max": aov_max,
        "ANOVA_min": aov_min,
        "ANOVA_range": aov_range,
        "timepoint_used": meta_df,
        "long_mean": long_mean,
        "long_max": long_max,
        "long_min": long_min,
        "long_range": long_range,
    }

    for label, aov in timepoint_aovs.items():
        sheets[f"ANOVA_{label}"] = aov

    for label, long_df in timepoint_longs.items():
        sheets[f"long_{label}"] = long_df

    # Keep a compact overview sheet for Streamlit preview.
    overview_rows = [
        {"dv": "mean", "n_rows_long": len(long_mean)},
        {"dv": "max", "n_rows_long": len(long_max)},
        {"dv": "min", "n_rows_long": len(long_min)},
        {"dv": "range", "n_rows_long": len(long_range)},
    ]
    for label, long_df in timepoint_longs.items():
        overview_rows.append({"dv": label, "n_rows_long": len(long_df)})
    sheets["overview"] = pd.DataFrame(overview_rows)

    return sheets


def anova_sheets_to_excel_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    """Write ANOVA sheets to an in-memory .xlsx file."""
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        # Put overview first if present, then the rest in insertion order.
        if "overview" in sheets:
            sheets["overview"].to_excel(writer, sheet_name="overview", index=False)
        for sheet_name, df in sheets.items():
            if sheet_name == "overview":
                continue
            # Excel sheet names must be <=31 chars.
            safe_name = str(sheet_name)[:31]
            df.to_excel(writer, sheet_name=safe_name, index=False)

    buffer.seek(0)
    return buffer.getvalue()


def default_anova_targets(condition: str) -> dict[str, float]:
    """Default target timepoints used by the app for each condition."""
    if condition == "2s":
        return {"t0p5": 0.5, "t1p5": 1.5}
    if condition == "6s":
        return {"t0p5": 0.5, "t3": 3.0}
    return {"t0p5": 0.5}
