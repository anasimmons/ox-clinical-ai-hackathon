"""
Wearable Clinical Pipeline Backend
Processes .cwa, .gz, and .csv wearable data files to generate clinical snapshots with FHIR validation.
"""

import os
import re
import tempfile
import traceback
from pathlib import Path

from flask import Flask, request, jsonify
from flask_cors import CORS

from processor import (
    build_fhir_bundle,
    generate_clinical_snapshot,
    load_default_patient,
    process_wearable_file as _process_wearable_file,
)

BASE_DIR = Path(__file__).resolve().parent

# Load environment variables and initialize Flask app
app = Flask(__name__)
CORS(app)


def _parse_snapshot_sections(text: str) -> dict:
    """Parse snapshot text into a dict of {SECTION_NAME: content}."""
    headers = ['PATIENT', 'MONITORING PERIOD', 'KEY FINDINGS', 'CLINICAL INTERPRETATION', 'RECOMMENDATION']
    pattern = re.compile(
        r'^(' + '|'.join(re.escape(h) for h in headers) + r')\s*\|?\s*',
        re.IGNORECASE | re.MULTILINE
    )
    sections = {}
    current_key = None
    current_lines = []

    for line in text.splitlines():
        m = pattern.match(line.strip())
        if m:
            if current_key:
                sections[current_key] = '\n'.join(current_lines).strip()
            current_key = m.group(1).upper()
            rest = line.strip()[m.end():].strip()
            current_lines = [rest] if rest else []
        elif current_key is not None:
            current_lines.append(line.strip())

    if current_key:
        sections[current_key] = '\n'.join(current_lines).strip()

    return sections


def _metric_card(value, label, color, alert=False) -> str:
    bg = '#fff5f5' if alert else '#f7fafc'
    border = '#fed7d7' if alert else '#e2e8f0'
    val_color = '#c53030' if alert else color
    return (
        f'<div style="background:{bg};border:1px solid {border};border-radius:8px;'
        f'padding:16px;text-align:center;">'
        f'<span style="display:block;font-size:1.8em;font-weight:700;color:{val_color};">{value}</span>'
        f'<span style="font-size:0.85em;color:#718096;margin-top:4px;display:block;">{label}</span>'
        f'</div>'
    )


def _bullets_to_html(text: str) -> str:
    """Convert newline-separated bullet points to an HTML list."""
    lines = [l.lstrip('- ').strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return ''
    items = ''.join(f'<li style="margin-bottom:6px;">{l}</li>' for l in lines)
    return f'<ul style="margin:0;padding-left:1.2em;color:#4a5568;line-height:1.7;">{items}</ul>'


def render_pipeline_output(patient: dict, snapshot_text: str, bundle=None) -> str:
    """Render full pipeline output as a self-contained HTML page."""
    steps       = patient.get('mean_daily_steps')
    hr          = patient.get('mean_heart_rate_bpm')
    rhr         = patient.get('resting_hr_bpm')
    sedentary   = patient.get('mean_daily_sedentary_minutes')
    active_min  = patient.get('total_active_minutes')
    days_target = patient.get('days_meeting_150min_activity_target', 0)

    low_activity = steps is not None and steps < 5000
    color = '#d9534f' if low_activity else '#0275d8'

    rec_start = patient.get('recording_start', '')[:10]
    rec_end   = patient.get('recording_end', '')[:10]
    period    = rec_start if rec_start == rec_end else f'{rec_start} – {rec_end}'

    # --- metrics grid ---
    metrics_html = '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:20px 0;">'
    if steps is not None:
        metrics_html += _metric_card(f'{steps:,}', 'Daily Steps', color, alert=low_activity)
    if hr is not None:
        metrics_html += _metric_card(f'{hr} bpm', 'Mean HR', color)
    if rhr is not None:
        metrics_html += _metric_card(f'{rhr} bpm', 'Resting HR', color)
    if sedentary is not None:
        metrics_html += _metric_card(f'{sedentary} min', 'Daily Sedentary', color)
    if active_min is not None:
        metrics_html += _metric_card(f'{active_min} min', 'Total Active', color)
    metrics_html += _metric_card(
        str(days_target),
        'Days Meeting Activity Target',
        color,
        alert=(days_target == 0)
    )
    metrics_html += '</div>'

    # --- clinical snapshot sections ---
    sections = _parse_snapshot_sections(snapshot_text)
    section_config = {
        'KEY FINDINGS':            ('📊', 'Key Findings'),
        'CLINICAL INTERPRETATION': ('🔍', 'Clinical Interpretation'),
        'RECOMMENDATION':          ('💡', 'Recommendation'),
    }
    sections_html = ''
    for key, (icon, label) in section_config.items():
        content = sections.get(key, '')
        if not content:
            continue
        body = _bullets_to_html(content) or f'<p style="color:#4a5568;margin:0;">{content}</p>'
        sections_html += f'''
        <div style="background:white;border-radius:8px;padding:20px;
                    border-left:4px solid {color};box-shadow:0 2px 8px rgba(0,0,0,0.05);margin-bottom:16px;">
            <div style="font-weight:600;color:#2d3748;font-size:1.05em;margin-bottom:10px;">{icon} {label}</div>
            {body}
        </div>'''

    # --- FHIR observations table ---
    fhir_badge = ''
    fhir_section_html = ''
    if bundle is not None:
        entries = bundle.entry or []
        obs_count = len(entries)
        fhir_badge = (
            f'<span style="background:#c6f6d5;color:#276749;padding:4px 12px;border-radius:12px;'
            f'font-size:0.85em;font-weight:600;">✓ FHIR R4 · {obs_count} Observations</span>'
        )
        rows = ''
        for entry in entries:
            obs = entry.resource
            label = obs.code.text or '—'
            loinc = obs.code.coding[0].code if obs.code.coding else '—'
            vq    = obs.valueQuantity
            value = f'{vq.value:g} {vq.unit}' if vq else '—'
            rows += (
                f'<tr>'
                f'<td style="padding:8px 12px;border-bottom:1px solid #e2e8f0;">{label}</td>'
                f'<td style="padding:8px 12px;border-bottom:1px solid #e2e8f0;font-weight:600;">{value}</td>'
                f'<td style="padding:8px 12px;border-bottom:1px solid #e2e8f0;color:#718096;font-size:0.85em;">'
                f'<a href="https://loinc.org/{loinc}" target="_blank" '
                f'style="color:#3182ce;text-decoration:none;">{loinc}</a></td>'
                f'</tr>'
            )
        fhir_section_html = f'''
    <div class="card">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;">
            <div class="section-title">🔬 FHIR R4 Observations</div>
            {fhir_badge}
        </div>
        <table style="width:100%;border-collapse:collapse;font-size:0.95em;">
            <thead>
                <tr style="background:#f7fafc;">
                    <th style="padding:8px 12px;text-align:left;color:#4a5568;font-weight:600;border-bottom:2px solid #e2e8f0;">Metric</th>
                    <th style="padding:8px 12px;text-align:left;color:#4a5568;font-weight:600;border-bottom:2px solid #e2e8f0;">Value</th>
                    <th style="padding:8px 12px;text-align:left;color:#4a5568;font-weight:600;border-bottom:2px solid #e2e8f0;">LOINC</th>
                </tr>
            </thead>
            <tbody>{rows}</tbody>
        </table>
    </div>'''

    urgent_badge = (
        '<span style="background:#fed7d7;color:#c53030;padding:4px 12px;border-radius:12px;'
        'font-size:0.8em;font-weight:600;border:2px solid #feb2b2;margin-left:10px;">⚠ LOW ACTIVITY</span>'
        if low_activity else ''
    )

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Clinical Report – {patient.get("name", "")}</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, #f0f4f8 0%, #d9e2ec 100%);
            min-height: 100vh;
            padding: 32px 16px;
            line-height: 1.6;
        }}
        .page {{ max-width: 900px; margin: 0 auto; }}
        .card {{
            background: white;
            border-radius: 16px;
            padding: 32px;
            box-shadow: 0 4px 24px rgba(0,0,0,0.08);
            margin-bottom: 20px;
        }}
        .header-title {{
            font-size: 1.6em;
            font-weight: 700;
            color: #1a365d;
            display: flex;
            align-items: center;
            flex-wrap: wrap;
            gap: 8px;
        }}
        .patient-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 12px;
            margin-top: 16px;
        }}
        .patient-field span:first-child {{
            font-size: 0.75em;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #718096;
            display: block;
        }}
        .patient-field span:last-child {{
            font-weight: 600;
            color: #2d3748;
        }}
        .section-title {{
            font-size: 1.1em;
            font-weight: 600;
            color: #2d3748;
            margin-bottom: 4px;
        }}
        .divider {{
            border: none;
            border-top: 1px solid #e2e8f0;
            margin: 20px 0;
        }}
        .footer {{
            text-align: center;
            color: #a0aec0;
            font-size: 0.82em;
            padding-top: 8px;
        }}
    </style>
</head>
<body>
<div class="page">

    <!-- Patient header -->
    <div class="card">
        <div class="header-title">
            🏥 Clinical Activity Report{urgent_badge}
        </div>
        <div class="patient-grid" style="margin-top:20px;">
            <div class="patient-field">
                <span>Patient</span>
                <span>{patient.get("name", "—")}</span>
            </div>
            <div class="patient-field">
                <span>Subject ID</span>
                <span>{patient.get("subject_id", "—")}</span>
            </div>
            <div class="patient-field">
                <span>Date of Birth</span>
                <span>{patient.get("dob", "—")}</span>
            </div>
            <div class="patient-field">
                <span>Device</span>
                <span>{patient.get("device", "—")}</span>
            </div>
            <div class="patient-field">
                <span>Recording Period</span>
                <span>{period}</span>
            </div>
            <div class="patient-field">
                <span>Clinical Context</span>
                <span>{patient.get("clinical_context", "—")}</span>
            </div>
        </div>
    </div>

    <!-- Metrics -->
    <div class="card">
        <div class="section-title">📈 Activity Metrics</div>
        {metrics_html}
    </div>

    <!-- Clinical snapshot -->
    <div class="card">
        <div class="section-title" style="margin-bottom:16px;">🩺 Clinical Snapshot</div>
        {sections_html}
    </div>

    {fhir_section_html}

    <div class="footer">Clinical AI Pipeline · GPT-4o-mini · FHIR R4</div>
</div>
</body>
</html>'''


@app.route('/', methods=['GET'])
def index():
    cwa_path = BASE_DIR / os.getenv('CWA_FILE', 'data/tiny-sample.cwa')
    subject_metadata = load_default_patient(os.getenv('PATIENT_FILE', 'PID-20394.json'))
    patient = _process_wearable_file(str(cwa_path), subject_metadata)
    bundle = build_fhir_bundle(patient)
    snapshot_text = generate_clinical_snapshot(patient, bundle)
    return render_pipeline_output(patient, snapshot_text, bundle)


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'service': 'Wearable Clinical Pipeline'})


@app.route('/process', methods=['POST'])
def process_wearable_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in ['.cwa', '.gz']:
            return jsonify({'error': f'Unsupported file format: {file_ext}. Accepted: .cwa, .gz'}), 400

        with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        try:
            subject_metadata = load_default_patient()
            patient = _process_wearable_file(tmp_path, subject_metadata)
            bundle = build_fhir_bundle(patient)
            snapshot = generate_clinical_snapshot(patient, bundle)
            return render_pipeline_output(patient, snapshot, bundle)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error processing file: {error_trace}")
        return jsonify({'error': str(e), 'traceback': error_trace}), 500


@app.route('/snapshot', methods=['POST'])
def generate_snapshot_endpoint():
    try:
        patient = request.get_json()
        if not patient:
            return jsonify({'error': 'No patient data provided'}), 400

        snapshot = generate_clinical_snapshot(patient)
        return jsonify({'status': 'success', 'clinical_snapshot': snapshot}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5001)
