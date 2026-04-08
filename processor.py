"""Wearable clinical data processing utilities."""

import json
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from fhir.resources.bundle import Bundle
from challenge_data import PATIENTS, to_pipeline_format

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

client = OpenAI()

SNAPSHOT_SYS_FT = (BASE_DIR / 'prompts' / 'snapshot_system_prompt.txt').read_text(encoding='utf-8')
CLINICAL_GUIDELINES = (BASE_DIR / 'prompts' / 'clinical_guidelines.txt').read_text(encoding='utf-8')
FHIR_SYSTEM = (BASE_DIR / 'prompts' / 'fhir_system.txt').read_text(encoding='utf-8')


def load_all_patients() -> list[dict]:
    """Return all patients converted to the flat pipeline format."""
    return [to_pipeline_format(p) for p in PATIENTS]


# Physiological bounds: (min, max) — values outside these are data errors, not clinical signals
_METRIC_BOUNDS = {
    "mean_daily_steps":                   (0,      100_000),
    "total_active_minutes":               (0,      1_440),
    "mean_daily_sedentary_minutes":        (0,      1_440),
    "mean_heart_rate_bpm":                (20,     300),
    "resting_hr_bpm":                     (20,     200),
    "days_meeting_150min_activity_target": (0,      7),
    "wear_duration_hours":                (0,      168),   # max = 7 days
}


def validate_patient_metrics(patient: dict) -> dict:
    """
    Check each metric against physiological bounds.
    - Impossible values (negative steps, HR of 0) raise ValueError.
    - Values that exceed the plausible ceiling are clamped; a flag is recorded.
    Returns a copy of the patient dict with clamped values and a '_flags' list.
    """
    cleaned = dict(patient)
    flags = []

    for field, (lo, hi) in _METRIC_BOUNDS.items():
        value = cleaned.get(field)
        if value is None:
            continue

        if value < lo:
            raise ValueError(
                f"[{patient.get('subject_id')}] '{field}' = {value} is below the "
                f"physiological minimum of {lo}. This is likely a data error."
            )

        if value > hi:
            flags.append({
                "field":    field,
                "original": value,
                "clamped":  hi,
                "message":  (
                    f"{field.replace('_', ' ').title()} value of {value} "
                    f"exceeds plausible maximum — clamped to {hi}"
                ),
            })
            cleaned[field] = hi

    cleaned["_flags"] = flags
    return cleaned


def build_fhir_bundle(patient: dict) -> Bundle:
    """
    Use an LLM to build a FHIR R4 Bundle from a patient metrics dict.
    Expects patient dict to have already been passed through validate_patient_metrics.
    Validation flags are included in the prompt so the model is aware of clamped values.
    """
    fhir_prompt = FHIR_SYSTEM
    if patient.get("_flags"):
        flags_text = "\n".join(f["message"] for f in patient["_flags"])
        fhir_prompt += f"\n\nData quality notes (values were clamped before submission):\n{flags_text}"

    r = client.chat.completions.create(
        model='gpt-4o-mini',
        response_format={'type': 'json_object'},
        messages=[
            {'role': 'system', 'content': fhir_prompt},
            {'role': 'user', 'content': json.dumps(patient)},
        ],
        temperature=0.0,
    )
    fhir_json = json.loads(r.choices[0].message.content)
    bundle = Bundle.model_validate(fhir_json)
    return bundle


def generate_clinical_snapshot(patient: dict, bundle=None) -> str:
    """Generate clinical snapshot text from patient metrics and FHIR bundle."""
    eval_prompt = f"{SNAPSHOT_SYS_FT}\n\nClinical Guidelines to follow:\n{CLINICAL_GUIDELINES}"

    # Build user content: raw metrics + structured FHIR observations
    user_content = {"patient_metrics": patient}
    if bundle is not None:
        observations = []
        for entry in (bundle.entry or []):
            obs = entry.resource
            if not hasattr(obs, 'code') or obs.code is None:
                continue
            code = obs.code.text if obs.code else "Unknown"
            vq = obs.valueQuantity
            observations.append({
                "metric": code,
                "value": float(vq.value),
                "unit": vq.unit,
                "loinc": obs.code.coding[0].code if obs.code.coding else None,
            })
        user_content["fhir_observations"] = observations

    response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[
            {'role': 'system', 'content': eval_prompt},
            {'role': 'user', 'content': json.dumps(user_content)}
        ],
        temperature=0.1
    )
    return response.choices[0].message.content
