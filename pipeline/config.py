from dataclasses import dataclass, field


@dataclass
class ProcessingConfig:
    """Settings copied from the verified notebook workflow."""

    # Columns
    time_col: str = "Application Time"
    pupil_left_col: str = "Pupil Radius (millimeters) left"
    pupil_right_col: str = "Pupil Radius (millimeters) right"
    eye_state_left: str = "Eye State left"
    eye_state_right: str = "Eye State right"

    # Filtering settings
    delta_threshold: float = 0.05
    fs: float = 70.0
    lp_order: int = 3
    lp_cutoff: float = 3.0

    # IMPORTANT:
    # The uploaded notebook text calls mark_invalid_pupil_samples(..., neighbor_radius=3),
    # but the uploaded processed_sub15_2s_stimuli1.csv is reproduced exactly only with radius=1.
    # Keep this as 1 if the goal is to reproduce the provided processed CSV.
    neighbor_radius: int = 1

    # Cycle skip thresholds
    max_bad_cycles_2s: int = 15
    max_bad_cycles_6s: int = 5

    # Naming convention for average_stimuliN.ipynb outputs.
    # 2s / 6s are classified at the raw CSV stage and remain separate afterward.
    # Therefore:
    # block2s_fold_mean.csv -> 2sstimuli_means_1to5.csv
    # block6s_fold_mean.csv -> 6sstimuli_means_1to5.csv
    block2s_stimuli_output_label: str = "2s"
    block6s_stimuli_output_label: str = "6s"

    # Rule parameters from integrated_filter_fin_keep_blanked_cycles_nan.ipynb
    rules: dict = field(default_factory=lambda: {
        "2s": {
            "trim_start": 1.5,
            "trim_end": 0.5,
            "cycle_len": 2.0,
            "cycle_start": 1.5,
            "bad_ratio": 0.25,
            "fold_time_min": 1.5,
            "fold_time_max_exclusive": 89.6,
        },
        "6s": {
            "trim_start": 4.5,
            "trim_end": 1.5,
            "cycle_len": 6.0,
            "cycle_start": 4.5,
            "bad_ratio": 0.20,
            "fold_time_min": 4.5,
            "fold_time_max_exclusive": 88.6,
        },
    })

    @property
    def err_left_col(self) -> str:
        return f"{self.pupil_left_col} error"

    @property
    def err_right_col(self) -> str:
        return f"{self.pupil_right_col} error"

    @property
    def filtered_left_col(self) -> str:
        return "Pupil Radius_Filtered_left"

    def max_bad_cycles(self, rule_key: str | None) -> int | None:
        if rule_key == "2s":
            return int(self.max_bad_cycles_2s)
        if rule_key == "6s":
            return int(self.max_bad_cycles_6s)
        return None
