from pathlib import Path

import pandas as pd

from .config import ProcessingConfig


def build_merged_exact(paths: list[Path], config: ProcessingConfig) -> pd.DataFrame | None:
    """
    Exact app version of merge_all.ipynb::build_merged().

    Important: this intentionally inserts blank separator columns because the uploaded
    merge_all.ipynb does that. This is preserved so later scripts receive the same file shape.
    """
    paths = [Path(p) for p in paths]
    if not paths:
        return None

    df_time_src = pd.read_csv(paths[0], low_memory=False)
    if df_time_src.shape[1] < 1:
        raise ValueError(f"No columns found in input file: {paths[0]}")

    time_col_name = df_time_src.columns[0]
    time_series = df_time_src.iloc[:, 0]

    frames = []
    names = []
    max_len = len(df_time_src)

    for path in paths:
        df = pd.read_csv(path, usecols=[config.filtered_left_col])
        max_len = max(max_len, len(df))

        base = path.stem
        if base.startswith("processed_"):
            base = base[len("processed_"):]

        df = df.rename(columns={config.filtered_left_col: f"{base}_left"})
        frames.append(df)
        names.append(base)

    max_index = range(max_len)
    time_series = time_series.reindex(max_index).astype("object").fillna("")

    merged = pd.DataFrame({time_col_name: time_series})
    merged = pd.concat([merged, pd.DataFrame({"": [""] * len(max_index)})], axis=1)

    for i, (df_one, base) in enumerate(zip(frames, names)):
        df_one = df_one.reindex(max_index).astype("object").fillna("")
        merged = pd.concat([merged, df_one], axis=1)
        if i < len(frames) - 1:
            merged = pd.concat([merged, pd.DataFrame({"": [""] * len(max_index)})], axis=1)

    return merged


def split_processed_files(processed_files: list[Path]) -> tuple[list[Path], list[Path]]:
    paths_2s: list[Path] = []
    paths_6s: list[Path] = []

    for p in processed_files:
        base = Path(p).stem
        if base.startswith("processed_"):
            base = base[len("processed_"):]

        if "2s" in base:
            paths_2s.append(Path(p))
        elif "6s" in base:
            paths_6s.append(Path(p))

    return paths_2s, paths_6s


def merge_processed_left_exact(processed_files: list[Path], config: ProcessingConfig):
    """Return (merged_2s, merged_6s) using exact merge_all.ipynb behavior."""
    paths_2s, paths_6s = split_processed_files(processed_files)
    merged_2s = build_merged_exact(paths_2s, config)
    merged_6s = build_merged_exact(paths_6s, config)
    return merged_2s, merged_6s
