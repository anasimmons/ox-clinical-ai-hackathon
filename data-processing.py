import actipy
import numpy as np
import pandas as pd
import json
import os
from pathlib import Path
from datetime import timezone

# Set Java environment for actipy
os.environ['JAVA_HOME'] = '/usr/local/opt/openjdk'
os.environ['PATH'] = '/usr/local/opt/openjdk/bin:' + os.environ.get('PATH', '')

def process_cwa_to_summary(
    cwa_path: str,
    subject_metadata: dict,
    heart_rate_bpm: int | None = None,
    resting_hr_bpm: int | None = None,
) -> dict:
    """
    Process a .cwa file into a clinical activity summary.

    Parameters
    ----------
    cwa_path : str
        Path to the .cwa file.
    subject_metadata : dict
        Keys: subject_id, name, dob (YYYY-MM-DD), clinical_context, device
    heart_rate_bpm : int | None
        Mean HR if available from external source (e.g. ECG, Holter).
    resting_hr_bpm : int | None
        Resting HR if available from external source.

    Returns
    -------
    dict matching the clinical summary schema.
    """

    # ------------------------------------------------------------------ #
    # 1. Read and process the .cwa file
    # ------------------------------------------------------------------ #
    data, info = actipy.read_device(
        cwa_path,
        lowpass_hz=20,          # low-pass filter at 20 Hz
        calibrate_gravity=True,  # autocalibration (GGIR-style)
        detect_nonwear=True,     # mark non-wear periods as NaN
        resample_hz=50,         # resample to 50 Hz
    )

    # data is a pandas DataFrame with columns: x, y, z, temperature, light
    # Non-wear periods have NaN in x, y, z

    # ------------------------------------------------------------------ #
    # 2. Compute magnitude and activity flag (if not already present)
    # ------------------------------------------------------------------ #
    if "magnitude" not in data.columns:
        data["magnitude"] = np.sqrt(data["x"]**2 + data["y"]**2 + data["z"]**2)

    # ENMO (Euclidean Norm Minus One) — standard activity proxy
    data["enmo"] = np.maximum(data["magnitude"] - 1, 0)

    # ------------------------------------------------------------------ #
    # 3. Recording window
    # ------------------------------------------------------------------ #
    recording_start = data.index[0].to_pydatetime().replace(tzinfo=timezone.utc)
    recording_end   = data.index[-1].to_pydatetime().replace(tzinfo=timezone.utc)

    total_duration_days = (recording_end - recording_start).total_seconds() / 86400
    # Partial days: use floor for 'days' denominator to avoid inflating per-day averages
    full_days = max(int(total_duration_days), 1)

    # ------------------------------------------------------------------ #
    # 4. Resample to 1-minute epochs for activity classification
    # ------------------------------------------------------------------ #
    # Drop non-wear (NaN rows) before epoching
    valid = data.dropna(subset=["x", "y", "z"])

    epoch_1min = valid["enmo"].resample("1min").mean()

    # Activity thresholds (mg) — standard Hildebrand 2014 cut-points for wrist
    # Sedentary : ENMO < 0.030 g  →  < 30 mg
    # Light     : 0.030 ≤ ENMO < 0.100 g
    # MVPA      : ENMO ≥ 0.100 g  →  ≥ 100 mg
    SEDENTARY_THRESHOLD = 0.030  # g (ENMO)
    MVPA_THRESHOLD      = 0.100  # g

    sedentary_minutes = int((epoch_1min < SEDENTARY_THRESHOLD).sum())
    active_minutes    = int((epoch_1min >= MVPA_THRESHOLD).sum())   # MVPA minutes

    mean_daily_sedentary = round(sedentary_minutes / full_days)
    total_active_minutes = active_minutes  # total across recording, not per-day

    # ------------------------------------------------------------------ #
    # 5. 150-minute weekly target → convert to recording period
    # ------------------------------------------------------------------ #
    # WHO target: 150 min MVPA per week → ~21.4 min/day
    # Flag each calendar day that meets ≥21 min MVPA
    daily_active = (
        epoch_1min[epoch_1min >= MVPA_THRESHOLD]
        .resample("1D")
        .count()
        .reindex(pd.date_range(recording_start.date(), recording_end.date(), freq="D"), fill_value=0)
    )
    DAILY_TARGET_MINUTES = 150 / 7  # ≈ 21.4
    days_meeting_target = int((daily_active >= DAILY_TARGET_MINUTES).sum())

    # ------------------------------------------------------------------ #
    # 6. Step estimation
    # ------------------------------------------------------------------ #
    # actipy doesn't include a step counter natively; use peak detection
    # on the band-passed vertical (z) axis. Band-pass 0.5–3 Hz captures
    # the 1–2 Hz cadence of normal walking; each peak = one heel-strike.
    from scipy.signal import butter, sosfilt, find_peaks

    sample_rate = info.get("ResampleRate", 50)  # Hz

    sos = butter(4, [0.5, 3.0], btype="bandpass", fs=sample_rate, output="sos")
    z_filtered = sosfilt(sos, valid["z"].fillna(0).values)

    # Steps can't be closer than 0.3 s apart (~200 steps/min max)
    min_distance_samples = int(0.3 * sample_rate)
    peaks, _ = find_peaks(z_filtered, height=0.1, distance=min_distance_samples)

    total_steps      = len(peaks)
    mean_daily_steps = round(total_steps / full_days)

    # ------------------------------------------------------------------ #
    # 7. Assemble output
    # ------------------------------------------------------------------ #
    summary = {
        # --- identity (from caller) ---
        "subject_id":       subject_metadata.get("subject_id"),
        "name":             subject_metadata.get("name"),
        "dob":              subject_metadata.get("dob"),
        "clinical_context": subject_metadata.get("clinical_context"),

        # --- device (from .cwa metadata via actipy info dict) ---
        "device": subject_metadata.get("device") or info.get("DeviceID", "Unknown"),

        # --- recording window ---
        "recording_start": recording_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "recording_end":   recording_end.strftime("%Y-%m-%dT%H:%M:%SZ"),

        # --- activity metrics ---
        "mean_daily_steps":               mean_daily_steps,
        "total_active_minutes":           total_active_minutes,
        "mean_daily_sedentary_minutes":   mean_daily_sedentary,
        "days_meeting_150min_activity_target": days_meeting_target,

        # --- heart rate (external source only — AX3 has no HR sensor) ---
        "mean_heart_rate_bpm": heart_rate_bpm,
        "resting_hr_bpm":      resting_hr_bpm,
    }

    return summary


# ------------------------------------------------------------------ #
# Example usage
# ------------------------------------------------------------------ #
if __name__ == "__main__":

    # Load metadata from patients/default_patient.json
    metadata_path = Path("patients/default_patient.json")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    result = process_cwa_to_summary(
        cwa_path="data/tiny-sample.cwa",
        subject_metadata=metadata,
        heart_rate_bpm=72,    # from Holter/ECG if available, else omit
        resting_hr_bpm=62,
    )

    print(json.dumps(result, indent=2))