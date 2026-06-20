# Pupil Diameter Analysis App — 2s / 6s separated outputs + rmANOVA

This app takes input of raw pupil datas from Fove headset, outputs teh filtered/organized/analyed datas.

## Output structure

The downloaded ZIP is organized as follows:

```text
2s/
├── processed/
│   └── processed_*.csv
├── block2s_fold_mean.csv
├── 2sstimuli_means_1to5.csv
└── anova/
    └── rmANOVA_2s.xlsx

6s/
├── processed/
│   └── processed_*.csv
├── block6s_fold_mean.csv
├── 6sstimuli_means_1to5.csv
└── anova/
    └── rmANOVA_6s.xlsx

qc_summary.csv
```

The requested three outputs per condition are:

1. `processed/` folder containing individual processed files
2. `blockNs_fold_mean.csv`
3. `Nsstimuli_means_1to5.csv`

The ANOVA Excel file is generated as an additional output under `anova/` when ANOVA is enabled.

## ANOVA behavior

The ANOVA module takes `block2s_fold_mean.csv` or `block6s_fold_mean.csv` as input and computes repeated-measures ANOVA by stimulus for:

- mean
- selected relative timepoints
- max
- min
- range

Default target timepoints:

```text
2s: 0.5s and 1.5s
6s: 0.5s and 3.0s
```

These can be changed from the sidebar.

## How to execute

```bash
cd /Users/satoumasatoshi/Desktop/pupillaryDiameterAnalysis-app
source .venv/bin/activate
python3 -m pip install streamlit
pip install -r requirements.txt
streamlit run app.py
```
