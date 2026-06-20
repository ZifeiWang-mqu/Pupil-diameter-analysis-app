from io import BytesIO
import zipfile


def create_zip(files_dict: dict[str, str | bytes]) -> BytesIO:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for filename, data in files_dict.items():
            if isinstance(data, str):
                data = data.encode("utf-8")
            zip_file.writestr(filename, data)
    buffer.seek(0)
    return buffer


def drop_blank_separator_columns(df):
    """Return a copy without merge_all-style blank separator columns.

    This is used for Streamlit preview and for in-memory fold averaging.
    The exact CSV export can still keep blank separator columns.
    """
    import pandas as pd

    if df is None:
        return None

    keep_cols = []
    for col in df.columns:
        col_str = str(col)
        if col_str.strip() == "":
            continue
        if col_str.startswith("Unnamed"):
            continue
        keep_cols.append(col)

    return df.loc[:, keep_cols].copy()


def make_streamlit_safe_df(df):
    """Make DataFrame column names unique for st.dataframe / pyarrow.

    Streamlit uses pyarrow internally. pyarrow rejects duplicate column names,
    while merge_all.ipynb intentionally creates repeated blank column names.
    This function is only for display; do not use it for exact CSV export.
    """
    if df is None:
        return None

    out = df.copy()
    counts = {}
    new_cols = []

    for i, col in enumerate(out.columns):
        label = str(col)
        if label.strip() == "":
            label = f"blank_separator_{i}"

        if label in counts:
            counts[label] += 1
            label = f"{label}_{counts[label]}"
        else:
            counts[label] = 0

        new_cols.append(label)

    out.columns = new_cols
    return out
