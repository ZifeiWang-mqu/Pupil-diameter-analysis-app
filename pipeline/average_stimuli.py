import numpy as np
import pandas as pd


def average_stimuliN_from_block_average(
    df: pd.DataFrame,
    *,
    stim_numbers=range(1, 6),
    time_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Exact app version of average_stimuliN.ipynb.

    Input
    -----
    df
        Output DataFrame from block2s_average_from_file.ipynb or
        block6s_average_from_file.ipynb.

    Logic copied from average_stimuliN.ipynb:
    - first column is treated as time
    - all other columns are numeric-converted
    - for each stimuliN, collect columns whose column name contains "stimuliN"
    - compute row-wise mean across those columns with skipna=True
    - output time column + stimuli1_mean ... stimuli5_mean

    Returns
    -------
    out
        Time column plus stimuliN_mean columns.
    group_info
        Table showing how many input columns were used for each stimuli group.
    """
    if df is None or df.empty:
        raise ValueError("Input block-average DataFrame is empty.")

    df = df.copy()

    # Notebook behavior: the first column is the time column.
    time_col = time_col or df.columns[0]

    if time_col not in df.columns:
        raise ValueError(f"Missing time column: {time_col}")

    data_cols = [c for c in df.columns if c != time_col]

    # Same as notebook: convert all non-time columns to numeric.
    # Blank separator columns become NaN and are ignored unless their name contains stimuliN.
    df_num = df[data_cols].apply(pd.to_numeric, errors="coerce")
    time_series = df[time_col]

    out = pd.DataFrame({time_col: time_series})
    group_info_rows = []

    for n in stim_numbers:
        key = f"stimuli{n}"
        cols_n = [c for c in data_cols if key in str(c)]
        group_info_rows.append({"group": key, "n_columns": len(cols_n)})

        if len(cols_n) == 0:
            out[f"{key}_mean"] = np.nan
        else:
            out[f"{key}_mean"] = df_num[cols_n].mean(axis=1, skipna=True)

    group_info = pd.DataFrame(group_info_rows)
    return out, group_info


def make_stimuli_means_filename(label: str) -> str:
    """
    Notebook/output examples use names like: 6sstimuli_means_1to5.csv.
    label should be '2s' or '6s'.
    """
    label = str(label).strip()
    return f"{label}stimuli_means_1to5.csv"
