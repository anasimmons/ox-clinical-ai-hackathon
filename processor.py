"""Wearable clinical data processing utilities."""

import importlib.util
import os
import json
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from fhir.resources.bundle import Bundle

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
os.environ['JAVA_HOME'] = '/usr/local/opt/openjdk'
os.environ['PATH'] = '/usr/local/opt/openjdk/bin:' + os.environ.get('PATH', '')

# Import process_cwa_to_summary from data-processing.py (hyphen prevents normal import)
_spec = importlib.util.spec_from_file_location("data_processing", BASE_DIR / "data-processing.py")
_dp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_dp)
process_cwa_to_summary = _dp.process_cwa_to_summary

client = OpenAI()

SNAPSHOT_SYS_FT = (BASE_DIR / 'prompts' / 'snapshot_system_prompt.txt').read_text(encoding='utf-8')
CLINICAL_GUIDELINES = (BASE_DIR / 'prompts' / 'clinical_guidelines.txt').read_text(encoding='utf-8')
DEFAULT_PATIENT_FILE = BASE_DIR / 'patients' / 'default_patient.json'
FHIR_SYSTEM = (BASE_DIR / 'prompts' / 'fhir_system.txt').read_text(encoding='utf-8')

def load_default_patient(filename: str = 'default_patient.json') -> dict:
    path = BASE_DIR / 'patients' / filename
    if not path.exists():
        raise FileNotFoundError(f"Default patient file not found: {path}")
    return json.loads(path.read_text(encoding='utf-8'))


def process_wearable_file(file_path: str, subject_metadata: dict,
                          heart_rate_bpm: int | None = None,
                          resting_hr_bpm: int | None = None) -> dict:
    """Process a .cwa/.gz wearable file and return a clinical summary dict."""
    file_ext = Path(file_path).suffix.lower()
    if file_ext not in ('.cwa', '.gz'):
        raise ValueError(f"Unsupported file format: {file_ext}. Accepted: .cwa, .gz")
    return process_cwa_to_summary(file_path, subject_metadata,
                                  heart_rate_bpm=heart_rate_bpm,
                                  resting_hr_bpm=resting_hr_bpm)


def build_fhir_bundle(patient: dict) -> Bundle:
    """
    Deterministically build a FHIR R4 Bundle from a patient metrics dict.
    Returns a fhir.resources Bundle object (no LLM involved).
    """
    from fhir.resources.bundle import Bundle, BundleEntry
    from fhir.resources.observation import Observation
    from fhir.resources.reference import Reference
    from fhir.resources.period import Period

    subject_ref = Reference(**{"reference": f"Patient/{patient['subject_id']}"})
    period = Period(**{
        "start": patient.get("recording_start"),
        "end":   patient.get("recording_end"),
    })

    CATEGORY = [{
        "coding": [{
            "system": "http://terminology.hl7.org/CodeSystem/observation-category",
            "code":   "activity",
            "display": "Activity",
        }]
    }]

    # (display, loinc_code, value, unit, ucum_unit)
    metrics = [
        ("Mean Daily Steps",             "41950-7", patient.get("mean_daily_steps"),               "steps/day",  "{steps}/d"),
        ("Total Active Minutes",          "55423-8", patient.get("total_active_minutes"),            "min",         "min"),
        ("Mean Heart Rate",               "8867-4",  patient.get("mean_heart_rate_bpm"),             "bpm",         "/min"),
        ("Resting Heart Rate",            "8867-4",  patient.get("resting_hr_bpm"),                  "bpm",         "/min"),
        ("Mean Daily Sedentary Minutes",  "82291-6", patient.get("mean_daily_sedentary_minutes"),    "min/day",    "min/d"),
        ("Days Meeting Activity Target",  "68516-4", patient.get("days_meeting_150min_activity_target"), "days", "d"),
    ]

    entries = []
    for display, loinc, value, unit_display, ucum in metrics:
        if value is None:
            continue
        obs = Observation(**{
            "status": "final",
            "category": CATEGORY,
            "code": {
                "coding": [{"system": "http://loinc.org", "code": loinc, "display": display}],
                "text": display,
            },
            "subject": {"reference": subject_ref.reference},
            "effectivePeriod": {"start": period.start, "end": period.end},
            "valueQuantity": {
                "value": float(value),
                "unit":  unit_display,
                "system": "http://unitsofmeasure.org",
                "code":   ucum,
            },
        })
        entries.append(BundleEntry(**{
            "fullUrl": f"urn:uuid:{display.lower().replace(' ', '-')}",
            "resource": obs,
            "request": {"method": "POST", "url": "Observation"},
        }))

    return Bundle(**{"type": "transaction", "entry": entries})


def generate_clinical_snapshot(patient: dict, bundle=None) -> str:
    """Generate clinical snapshot text from patient metrics and FHIR bundle."""
    eval_prompt = f"{SNAPSHOT_SYS_FT}\n\nClinical Guidelines to follow:\n{CLINICAL_GUIDELINES}"

    # Build user content: raw metrics + structured FHIR observations
    user_content = {"patient_metrics": patient}
    if bundle is not None:
        observations = []
        for entry in (bundle.entry or []):
            obs = entry.resource
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
