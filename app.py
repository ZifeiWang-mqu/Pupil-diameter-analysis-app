import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from pipeline.config import ProcessingConfig
from pipeline.preprocess import run_pipeline, detect_rule_key
from pipeline.merge import merge_processed_left_exact
from pipeline.fold_average import block2s_average_from_merged, block6s_average_from_merged
from pipeline.average_stimuli import average_stimuliN_from_block_average, make_stimuli_means_filename
from pipeline.anova import (
    anova_sheets_to_excel_bytes,
    default_anova_targets,
    run_rm_anova_from_block_average,
)
from pipeline.utils import create_zip, make_streamlit_safe_df, drop_blank_separator_columns


st.set_page_config(page_title="Pupil Diameter Analysis App", layout="wide")
st.title("Pupil Diameter Analysis App")

st.markdown(
    """

**Output structure inside the ZIP**

```text
2s/
├── processed/
├── block2s_fold_mean.csv
├── 2sstimuli_means_1to5.csv
└── anova/rmANOVA_2s.xlsx  # if ANOVA is enabled

6s/
├── processed/
├── block6s_fold_mean.csv
├── 6sstimuli_means_1to5.csv
└── anova/rmANOVA_6s.xlsx  # if ANOVA is enabled
```
"""
)

st.sidebar.header("Processing Settings")

st.sidebar.markdown(
    """
These parameters control the preprocessing step that converts raw FOVE CSV files into
`processed_*.csv` files. The default values are set to match the currently verified
notebook-based workflow.
"""
)

with st.sidebar.expander("Parameter guide", expanded=False):
    st.markdown(
        """
**Sampling rate / FS**  
Target resampling frequency in Hz. 
Default: `70 Hz`.

**Delta threshold**  
Sudden-change exclusion threshold for raw pupil radius before interpolation.
If the frame-to-frame pupil change is larger than this value, that sample is treated as invalid.
Default: `0.05 mm`.

**Low-pass filter order**  
Order of the Butterworth low-pass filter. Higher values create a steeper filter but can be less stable.
Default: `3`.

**Low-pass cutoff**  
Cutoff frequency for smoothing pupil radius after resampling. 
Default: `3.0 Hz`.

**Neighbor radius**  
Number of samples before and after an invalid sample that are also removed.

**Max bad cycles 2s / 6s**  
If the number of bad cycles in a file reaches this threshold, that file is removed.
Defaults: `2s = 15`, `6s = 5`.
"""
    )

fs_value = st.sidebar.number_input(
    "Sampling rate / FS",
    value=70.0,
    help="Target sampling frequency in Hz after resampling. The verified workflow uses 70 Hz.",
)

delta_threshold_value = st.sidebar.number_input(
    "Delta threshold",
    value=0.05,
    format="%.3f",
    help="Frame-to-frame pupil jump threshold in millimeters. Samples with abs(diff) greater than this are set to NaN before interpolation.",
)

lp_order_value = int(st.sidebar.number_input(
    "Low-pass filter order",
    value=3,
    step=1,
    help="Butterworth low-pass filter order. The original notebook uses order 3.",
))

lp_cutoff_value = st.sidebar.number_input(
    "Low-pass cutoff",
    value=3.0,
    help="Low-pass cutoff frequency in Hz. The original notebook uses 3.0 Hz.",
)

neighbor_radius_value = int(st.sidebar.number_input(
    "Neighbor radius",
    value=1,
    step=1,
    help="Number of neighboring samples on both sides of blink/low-accuracy/unreliable samples that are also set to NaN.",
))

max_bad_cycles_2s_value = int(st.sidebar.number_input(
    "Max bad cycles 2s",
    value=15,
    step=1,
    help="Skip a 2s file when n_bad_cycles is greater than or equal to this value.",
))

max_bad_cycles_6s_value = int(st.sidebar.number_input(
    "Max bad cycles 6s",
    value=5,
    step=1,
    help="Skip a 6s file when n_bad_cycles is greater than or equal to this value.",
))

config = ProcessingConfig(
    fs=fs_value,
    delta_threshold=delta_threshold_value,
    lp_order=lp_order_value,
    lp_cutoff=lp_cutoff_value,
    neighbor_radius=neighbor_radius_value,
    max_bad_cycles_2s=max_bad_cycles_2s_value,
    max_bad_cycles_6s=max_bad_cycles_6s_value,
)

st.sidebar.header("ANOVA Settings")
st.sidebar.markdown(
    """
The ANOVA step runs separately for the 2s and 6s outputs
"""
)

run_anova = st.sidebar.checkbox(
    "Run rmANOVA for 2s and 6s",
    value=True,
    help="When enabled, the app creates 2s/anova/rmANOVA_2s.xlsx and 6s/anova/rmANOVA_6s.xlsx from the block fold mean files.",
)

with st.sidebar.expander("ANOVA target times", expanded=False):
    st.caption(
        "Relative times are measured from the first row of each block fold mean file. "
        "The app uses the nearest available time sample."
    )
    anova_2s_t0 = st.number_input(
        "2s target time 1",
        value=0.5,
        format="%.6f",
        help="First timepoint extracted from block2s_fold_mean.csv for repeated-measures ANOVA. Default: 0.5 s.",
    )
    anova_2s_t1 = st.number_input(
        "2s target time 2",
        value=1.5,
        format="%.6f",
        help="Second timepoint extracted from block2s_fold_mean.csv for repeated-measures ANOVA. Default: 1.5 s.",
    )
    anova_6s_t0 = st.number_input(
        "6s target time 1",
        value=0.5,
        format="%.6f",
        help="First timepoint extracted from block6s_fold_mean.csv for repeated-measures ANOVA. Default: 0.5 s.",
    )
    anova_6s_t1 = st.number_input(
        "6s target time 2",
        value=3.0,
        format="%.6f",
        help="Second timepoint extracted from block6s_fold_mean.csv for repeated-measures ANOVA. Default: 3.0 s.",
    )

anova_targets_2s = {"t0p5": float(anova_2s_t0), "t1p5": float(anova_2s_t1)}
anova_targets_6s = {"t0p5": float(anova_6s_t0), "t3": float(anova_6s_t1)}

uploaded_files = st.file_uploader(
    "Upload raw CSV files",
    type=["csv"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info("Upload raw CSV files to start.")
    st.stop()

raw_2s = [f for f in uploaded_files if detect_rule_key(f.name) == "2s"]
raw_6s = [f for f in uploaded_files if detect_rule_key(f.name) == "6s"]
raw_unknown = [f for f in uploaded_files if detect_rule_key(f.name) is None]

st.success(f"{len(uploaded_files)} raw file(s) uploaded.")

class_col1, class_col2, class_col3 = st.columns(3)
with class_col1:
    st.metric("Raw 2s files", len(raw_2s))
with class_col2:
    st.metric("Raw 6s files", len(raw_6s))
with class_col3:
    st.metric("Unclassified files", len(raw_unknown))

with st.expander("Raw file classification", expanded=False):
    st.write("2s files")
    st.write([f.name for f in raw_2s] or "None")
    st.write("6s files")
    st.write([f.name for f in raw_6s] or "None")
    if raw_unknown:
        st.warning("Files without '2s' or '6s' in the filename will be skipped.")
        st.write([f.name for f in raw_unknown])

if not st.button("Run full separated pipeline", type="primary"):
    st.stop()

with tempfile.TemporaryDirectory() as tmpdir_str:
    tmpdir = Path(tmpdir_str)
    input_dir = tmpdir / "input"
    processed_dir = tmpdir / "processed"
    input_dir.mkdir()
    processed_dir.mkdir()

    summaries = []
    processed_files_2s: list[Path] = []
    processed_files_6s: list[Path] = []
    zip_files: dict[str, str | bytes] = {}

    progress = st.progress(0)

    for i, uploaded_file in enumerate(uploaded_files, start=1):
        rule_key = detect_rule_key(uploaded_file.name)
        input_path = input_dir / uploaded_file.name
        input_path.write_bytes(uploaded_file.getbuffer())

        if rule_key not in {"2s", "6s"}:
            summaries.append(
                {
                    "file": uploaded_file.name,
                    "rule": "unclassified",
                    "pre_interp_excl_left": float("nan"),
                    "cycle_blanked_rows_percent": float("nan"),
                    "n_cycles": float("nan"),
                    "n_bad_cycles": float("nan"),
                    "skip_threshold_bad_cycles": float("nan"),
                    "skipped": True,
                    "processed_output": None,
                    "output_group": "unclassified",
                    "error": "Filename must contain '2s' or '6s'.",
                }
            )
            progress.progress(i / len(uploaded_files))
            continue

        try:
            processed_df, summary, processed_path = run_pipeline(
                input_path,
                processed_dir,
                config,
            )
            summary["output_group"] = rule_key
            summaries.append(summary)

            if processed_path is not None:
                if rule_key == "2s":
                    processed_files_2s.append(processed_path)
                    zip_files[f"2s/processed/{processed_path.name}"] = processed_df.to_csv(index=False)
                else:
                    processed_files_6s.append(processed_path)
                    zip_files[f"6s/processed/{processed_path.name}"] = processed_df.to_csv(index=False)

        except Exception as e:
            summaries.append(
                {
                    "file": uploaded_file.name,
                    "rule": rule_key,
                    "pre_interp_excl_left": float("nan"),
                    "cycle_blanked_rows_percent": float("nan"),
                    "n_cycles": float("nan"),
                    "n_bad_cycles": float("nan"),
                    "skip_threshold_bad_cycles": float("nan"),
                    "skipped": True,
                    "processed_output": None,
                    "output_group": rule_key,
                    "error": str(e),
                }
            )

        progress.progress(i / len(uploaded_files))

    summary_df = pd.DataFrame(summaries).sort_values("file")
    zip_files["qc_summary.csv"] = summary_df.to_csv(index=False)

    st.subheader("1. Preprocessing QC Summary")
    st.dataframe(summary_df, use_container_width=True)

    all_processed_files = processed_files_2s + processed_files_6s
    merged_2s, merged_6s = merge_processed_left_exact(all_processed_files, config)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("2s output folder")
        st.caption("2s/processed/ + block2s_fold_mean.csv + 2sstimuli_means_1to5.csv")

        if merged_2s is None:
            st.info("No valid 2s processed files found.")
        else:
            with st.expander("Merged 2s preview (intermediate, not saved as final folder output)"):
                st.dataframe(make_streamlit_safe_df(merged_2s.head()), use_container_width=True)

            block2s = block2s_average_from_merged(drop_blank_separator_columns(merged_2s), config)
            zip_files["2s/block2s_fold_mean.csv"] = block2s.to_csv(index=False)

            st.write("`2s/block2s_fold_mean.csv`")
            st.dataframe(block2s.head(), use_container_width=True)

            stimuli2s, group_info2s = average_stimuliN_from_block_average(block2s)
            stimuli2s_filename = make_stimuli_means_filename("2s")
            zip_files[f"2s/{stimuli2s_filename}"] = stimuli2s.to_csv(index=False)

            st.write(f"`2s/{stimuli2s_filename}`")
            st.dataframe(stimuli2s.head(), use_container_width=True)

            with st.expander("Stimuli group counts for 2s output"):
                st.dataframe(group_info2s, use_container_width=True)

            if run_anova:
                try:
                    anova_sheets_2s = run_rm_anova_from_block_average(
                        block2s,
                        target_times_rel=anova_targets_2s,
                    )
                    anova_bytes_2s = anova_sheets_to_excel_bytes(anova_sheets_2s)
                    zip_files["2s/anova/rmANOVA_2s.xlsx"] = anova_bytes_2s
                    st.success("2s rmANOVA completed: `2s/anova/rmANOVA_2s.xlsx`")
                    st.dataframe(anova_sheets_2s["overview"], use_container_width=True)
                except Exception as e:
                    st.error(f"2s ANOVA failed: {e}")
                    zip_files["2s/anova/ANOVA_2s_ERROR.txt"] = str(e)

            chart_df = stimuli2s.copy()
            numeric_cols = [c for c in chart_df.columns[1:] if str(c).strip() != ""]
            if numeric_cols:
                st.line_chart(chart_df[[chart_df.columns[0]] + numeric_cols].set_index(chart_df.columns[0]))

    with col2:
        st.subheader("6s output folder")
        st.caption("6s/processed/ + block6s_fold_mean.csv + 6sstimuli_means_1to5.csv")

        if merged_6s is None:
            st.info("No valid 6s processed files found.")
        else:
            with st.expander("Merged 6s preview (intermediate, not saved as final folder output)"):
                st.dataframe(make_streamlit_safe_df(merged_6s.head()), use_container_width=True)

            block6s = block6s_average_from_merged(drop_blank_separator_columns(merged_6s), config)
            zip_files["6s/block6s_fold_mean.csv"] = block6s.to_csv(index=False)

            st.write("`6s/block6s_fold_mean.csv`")
            st.dataframe(block6s.head(), use_container_width=True)

            stimuli6s, group_info6s = average_stimuliN_from_block_average(block6s)
            stimuli6s_filename = make_stimuli_means_filename("6s")
            zip_files[f"6s/{stimuli6s_filename}"] = stimuli6s.to_csv(index=False)

            st.write(f"`6s/{stimuli6s_filename}`")
            st.dataframe(stimuli6s.head(), use_container_width=True)

            with st.expander("Stimuli group counts for 6s output"):
                st.dataframe(group_info6s, use_container_width=True)

            if run_anova:
                try:
                    anova_sheets_6s = run_rm_anova_from_block_average(
                        block6s,
                        target_times_rel=anova_targets_6s,
                    )
                    anova_bytes_6s = anova_sheets_to_excel_bytes(anova_sheets_6s)
                    zip_files["6s/anova/rmANOVA_6s.xlsx"] = anova_bytes_6s
                    st.success("6s rmANOVA completed: `6s/anova/rmANOVA_6s.xlsx`")
                    st.dataframe(anova_sheets_6s["overview"], use_container_width=True)
                except Exception as e:
                    st.error(f"6s ANOVA failed: {e}")
                    zip_files["6s/anova/ANOVA_6s_ERROR.txt"] = str(e)

            chart_df = stimuli6s.copy()
            numeric_cols = [c for c in chart_df.columns[1:] if str(c).strip() != ""]
            if numeric_cols:
                st.line_chart(chart_df[[chart_df.columns[0]] + numeric_cols].set_index(chart_df.columns[0]))

    st.download_button(
        label="Download output folders as ZIP",
        data=create_zip(zip_files),
        file_name="pupil_analysis_outputs_2s_6s_with_anova.zip",
        mime="application/zip",
    )
