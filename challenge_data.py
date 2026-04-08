"""
Wearables Challenge — Patient Data
Oxford Clinical AI Hackathon 2026 · Day 3 Demo Day · 8 April 2026

Three patient datasets for the wearables challenge.
Each represents 7 days of wrist-worn accelerometer + heart rate data
processed to summary form (post-actipy).

Data shape is the same as the `patient` dict in your Day 2 notebook,
enriched with daily breakdowns and one day of hourly HR data so you
can demonstrate deeper clinical reasoning.

USAGE
-----
Option 1 — Copy the PATIENTS list into a code cell in your notebook
Option 2 — Place this file alongside your notebook and import:

    from patient_data import PATIENTS
    mrs_whitfield = PATIENTS[0]
    mr_ahmed      = PATIENTS[1]
    mrs_prescott  = PATIENTS[2]

Each patient is a Python dict you can pass straight into your existing
FHIR transformation step and snapshot generation step.
"""

PATIENTS = [

    # ════════════════════════════════════════════════════════════
    # PATIENT 1
    # ════════════════════════════════════════════════════════════
    {
        "subject_id": "PID-2026-MW-04",
        "name": "Mrs Margaret Whitfield",
        "dob": "1961-09-12",
        "age": 64,
        "sex": "Female",
        "clinical_context": (
            "Post-MI cardiac rehabilitation, week 4 of 12. "
            "NSTEMI 26 February 2026, PCI to LAD with one drug-eluting stent. "
            "Discharge medications: aspirin 75 mg OD, ticagrelor 90 mg BD, "
            "atorvastatin 80 mg ON, bisoprolol 2.5 mg OD, ramipril 5 mg OD."
        ),
        "monitoring_period": {
            "start": "2026-04-01T00:00:00Z",
            "end":   "2026-04-07T23:59:59Z",
            "days":  7,
        },
        "metrics": {
            # ─── Weekly aggregates ───────────────────────────
            "steps_total":                45100,
            "steps_daily_avg":             6443,
            "active_minutes_total":         245,
            "active_minutes_daily_avg":      35,
            "heart_rate_resting_avg":        68,
            "heart_rate_max":               132,
            "sedentary_minutes_daily_avg":  480,
            "non_wear_minutes_daily_avg":    30,

            # ─── Daily breakdown (Mon → Sun) ─────────────────
            "steps_daily":         [5800, 6200, 6900, 6800, 7100, 6500, 5800],
            "rhr_daily":           [  70,   69,   68,   67,   68,   67,   68],
            "active_min_daily":    [  30,   32,   38,   40,   42,   35,   28],
            "sedentary_min_daily": [ 510,  495,  470,  460,  465,  475,  485],
            "non_wear_min_daily":  [  35,   28,   25,   30,   32,   28,   30],

            # ─── Hourly HR for a representative day (Wed, hours 00:00 → 23:00) ─
            "hr_hourly_representative_day": [
                62, 60, 58, 58, 58, 60,   # 00:00–05:00  sleep
                65, 72, 78, 85, 92, 95,   # 06:00–11:00  morning, breakfast, light activity
                88, 82, 78, 80, 85, 92,   # 12:00–17:00  lunch, afternoon walk
                88, 78, 72, 68, 65, 62,   # 18:00–23:00  evening, wind down
            ],
        },
        "context_notes": (
            "Attended both physiotherapy sessions this week. "
            "Patient self-reports feeling stronger and more confident with exercise. "
            "No chest pain. No breathlessness on exertion. Sleeping well."
        ),
    },

    # ════════════════════════════════════════════════════════════
    # PATIENT 2
    # ════════════════════════════════════════════════════════════
    {
        "subject_id": "PID-2026-TA-06",
        "name": "Mr Tariq Ahmed",
        "dob": "1954-11-30",
        "age": 71,
        "sex": "Male",
        "clinical_context": (
            "Post-MI cardiac rehabilitation, week 6 of 12. "
            "STEMI 24 January 2026, primary PCI to RCA. "
            "Background of paroxysmal atrial fibrillation on apixaban 5 mg BD. "
            "Other medications: aspirin 75 mg, atorvastatin 80 mg, bisoprolol 5 mg, ramipril 10 mg."
        ),
        "monitoring_period": {
            "start": "2026-04-01T00:00:00Z",
            "end":   "2026-04-07T23:59:59Z",
            "days":  7,
        },
        "metrics": {
            "steps_total":                36400,
            "steps_daily_avg":             5200,
            "active_minutes_total":         210,
            "active_minutes_daily_avg":      30,
            "heart_rate_resting_avg":        72,
            "heart_rate_max":               145,
            "sedentary_minutes_daily_avg":  540,
            "non_wear_minutes_daily_avg":   360,   # ~6 hrs/day

            "steps_daily":         [5400, 5100, 5300, 5200, 5500, 4900, 5000],
            "rhr_daily":           [  71,   72,   72,   73,   72,   71,   72],
            "active_min_daily":    [  32,   30,   28,   31,   30,   29,   30],
            "sedentary_min_daily": [ 535,  545,  540,  550,  530,  545,  555],
            "non_wear_min_daily":  [ 340,  365,  370,  355,  360,  380,  350],

            # Hourly HR for Thursday (hours 00:00 → 23:00)
            "hr_hourly_representative_day": [
                68, 66, 64, 132, 138, 145, # 00:00–05:00  episode of HR 130–145 between 03:00–05:00
                72, 75, 78,  80,  82,  85, # 06:00–11:00  back to baseline by 06:00, normal day
                78, 75, 72,  78,  82,  85, # 12:00–17:00
                80, 75, 72,  70,  68,  66, # 18:00–23:00
            ],
        },
        "context_notes": (
            "Attended one of two scheduled physiotherapy sessions this week "
            "(missed Friday — patient reported palpitations overnight). "
            "Last echocardiogram 4 weeks ago showed mild LV systolic dysfunction (EF 48%). "
            "No documented AF episodes in the past 6 months."
        ),
    },

    # ════════════════════════════════════════════════════════════
    # PATIENT 3
    # ════════════════════════════════════════════════════════════
    {
        "subject_id": "PID-2026-EP-02",
        "name": "Mrs Eleanor Prescott",
        "dob": "1948-05-18",
        "age": 78,
        "sex": "Female",
        "clinical_context": (
            "Post-MI cardiac rehabilitation, week 2 of 12. "
            "NSTEMI 18 March 2026, PCI to circumflex artery. "
            "Lives alone, widowed, daughter visits weekly. "
            "Background of mild bilateral knee osteoarthritis, type 2 diabetes "
            "(HbA1c 58 mmol/mol), well-controlled hypertension. "
            "Medications: aspirin, ticagrelor, atorvastatin, bisoprolol, ramipril, metformin."
        ),
        "monitoring_period": {
            "start": "2026-04-01T00:00:00Z",
            "end":   "2026-04-07T23:59:59Z",
            "days":  7,
        },
        "metrics": {
            "steps_total":                16800,
            "steps_daily_avg":             2400,
            "active_minutes_total":          84,
            "active_minutes_daily_avg":      12,
            "heart_rate_resting_avg":        84,    # weekly average — note rising trend in daily values
            "heart_rate_max":               108,
            "sedentary_minutes_daily_avg":  660,    # 11 hrs/day
            "non_wear_minutes_daily_avg":    45,

            # Daily — note RHR rises across the week (78 → 89)
            "steps_daily":         [3100, 2800, 2500, 2200, 2100, 2000, 2100],
            "rhr_daily":           [  78,   80,   82,   84,   86,   88,   89],
            "active_min_daily":    [  18,   15,   12,   11,    9,    9,   10],
            "sedentary_min_daily": [ 620,  640,  660,  670,  680,  680,  670],
            "non_wear_min_daily":  [  40,   42,   45,   48,   45,   50,   45],

            # Hourly HR for Sunday — persistently elevated, no nocturnal dip
            "hr_hourly_representative_day": [
                85, 84, 86, 85, 86, 87,  # 00:00–05:00  no proper nocturnal dip
                89, 92, 95, 96, 94, 92,  # 06:00–11:00
                90, 91, 93, 95, 92, 90,  # 12:00–17:00
                89, 91, 90, 88, 87, 86,  # 18:00–23:00
            ],
        },
        "context_notes": (
            "MISSED both physiotherapy appointments this week. "
            "Cardiac rehab nurse made two phone calls — reached patient on the second. "
            "Patient reported feeling 'tired and a bit short of breath' but declined to attend clinic. "
            "Daughter reports patient has been more withdrawn. "
            "No reported chest pain. No reported new medications."
        ),
    },

]


# ─── Optional helpers ───────────────────────────────────────────
# These do not change the data — they just make it easier to access
# from inside your notebook.

def get_patient_by_name(name):
    """Return the first patient whose `name` field contains the given string."""
    for p in PATIENTS:
        if name.lower() in p["name"].lower():
            return p
    return None


def all_patients():
    """Return the full list of three patient dicts."""
    return PATIENTS


def to_pipeline_format(p: dict) -> dict:
    """
    Convert a PATIENTS dict to the flat format expected by the pipeline
    (process_cwa_to_summary output schema), so the pre-processed patient
    data can be passed directly to build_fhir_bundle / generate_clinical_snapshot
    / render_pipeline_output without needing a .cwa file.
    """
    metrics = p["metrics"]
    period  = p["monitoring_period"]
    days    = period["days"]

    # Count days where MVPA minutes >= WHO daily target (150 min/week ÷ 7)
    DAILY_TARGET = 150 / 7  # ≈ 21.4 min
    days_meeting = sum(1 for m in metrics["active_min_daily"] if m >= DAILY_TARGET)

    # Wear hours = total recording hours minus average daily non-wear scaled to full period
    wear_hours = round(days * 24 - metrics["non_wear_minutes_daily_avg"] * days / 60, 1)

    # Best available mean HR proxy: average of the representative-day hourly values
    hr_hourly = metrics.get("hr_hourly_representative_day", [])
    mean_hr = round(sum(hr_hourly) / len(hr_hourly)) if hr_hourly else None

    return {
        "subject_id":       p["subject_id"],
        "name":             p["name"],
        "dob":              p["dob"],
        "clinical_context": p["clinical_context"],
        "device":           "Axivity AX3",
        "recording_start":  period["start"],
        "recording_end":    period["end"],
        "wear_duration_hours": wear_hours,
        "mean_daily_steps":               metrics["steps_daily_avg"],
        "total_active_minutes":           metrics["active_minutes_total"],
        "mean_daily_sedentary_minutes":   metrics["sedentary_minutes_daily_avg"],
        "days_meeting_150min_activity_target": days_meeting,
        "mean_heart_rate_bpm": mean_hr,
        "resting_hr_bpm":      metrics["heart_rate_resting_avg"],
    }


if __name__ == "__main__":
    # Quick sanity check — run `python patient_data.py` to confirm the file loads.
    for p in PATIENTS:
        print(f"{p['subject_id']:20s}  {p['name']:24s}  "
              f"week {p['clinical_context'].split('week ')[1].split(' ')[0]:>3s}  "
              f"steps/day {p['metrics']['steps_daily_avg']:>5d}  "
              f"RHR avg {p['metrics']['heart_rate_resting_avg']:>3d}")
