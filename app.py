"""
Wearable Clinical Pipeline Backend
Processes .cwa, .gz, and .csv wearable data files to generate clinical snapshots with FHIR validation.
"""

import os
import json
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np
import pandas as pd
import actipy
from fhir.resources.bundle import Bundle
from fhir.resources.patient import Patient
from fhir.resources.observation import Observation

# Load environment variables
load_dotenv()
app = Flask(__name__)
CORS(app)

# Set Java environment for actipy
os.environ['JAVA_HOME'] = '/usr/local/opt/openjdk'
os.environ['PATH'] = '/usr/local/opt/openjdk/bin:' + os.environ['PATH']

# Initialize OpenAI client
client = OpenAI()

# Load prompt files from disk
BASE_DIR = Path(__file__).resolve().parent

def load_text_file(filename: str) -> str:
    path = BASE_DIR / filename
    return path.read_text(encoding='utf-8')

SNAPSHOT_SYS_FT = load_text_file('snapshot_system_prompt.txt')
CLINICAL_GUIDELINES = load_text_file('clinical_guidelines.txt')

FHIR_SYSTEM = '''You are an expert in FHIR R4 interoperability.
Convert the wearable summary to a valid FHIR R4 Bundle (type: transaction).
Include one Observation per metric. Each Observation must have:
status:final, category:[{coding:[{system:http://terminology.hl7.org/CodeSystem/observation-category,code:activity}]}],
code with correct LOINC (steps:41950-7, active_min:55423-8, heart_rate:8867-4, sedentary:82291-6),
subject:{reference:Patient/{subject_id}}, effectivePeriod:{start,end},
valueQuantity:{value,unit,system:http://unitsofmeasure.org}.
Respond ONLY with valid JSON.'''


def load_wearable_data(file_path):
    """
    Load wearable data from .cwa, .gz, or .csv file.
    
    Args:
        file_path: Path to the wearable data file
        
    Returns:
        Tuple of (data DataFrame, info dict)
    """
    file_ext = Path(file_path).suffix.lower()
    
    if file_ext == '.cwa' or file_ext == '.gz':
        # Use actipy for .cwa and .gz files
        data, info = actipy.read_device(
            file_path,
            lowpass_hz=20,
            calibrate_gravity=True,
            detect_nonwear=True,
            resample_hz=50
        )
        return data, info
    elif file_ext == '.csv':
        # Load CSV with expected columns: timestamp, x, y, z, temperature, light
        data = pd.read_csv(file_path, parse_dates=['time'] if 'time' in pd.read_csv(file_path, nrows=1).columns else False)
        if 'time' not in data.columns and data.index.name == 'time':
            data = data.reset_index()
        data['time'] = pd.to_datetime(data['time'])
        data = data.set_index('time')
        info = {'Device': 'CSV Import'}
        return data, info
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")


def extract_patient_metrics(data, info):
    """
    Extract metrics from accelerometer data and create patient dictionary.
    
    Args:
        data: DataFrame from actipy.read_device()
        info: Device info dictionary
        
    Returns:
        Patient dictionary with clinical metrics
    """
    # Calculate acceleration magnitude for activity estimation
    data['magnitude'] = np.sqrt(data['x']**2 + data['y']**2 + data['z']**2)
    
    # Define activity threshold (high acceleration = active movement)
    activity_threshold = data['magnitude'].quantile(0.5)
    data['is_active'] = (data['magnitude'] > activity_threshold).astype(int)
    
    # Resample to 1-minute intervals for daily summaries
    data_1min = data.resample('1min').agg({
        'magnitude': 'mean',
        'is_active': 'max',
        'temperature': 'mean',
        'light': 'mean'
    }).reset_index()
    
    # Mark sedentary if inactive for extended periods
    data_1min['is_sedentary'] = (data_1min['is_active'] == 0).astype(int)
    
    # Create daily summary
    data_1min['date'] = data_1min['time'].dt.date
    daily = data_1min.groupby('date').agg({
        'is_active': 'sum',
        'is_sedentary': 'sum',
        'magnitude': 'mean'
    }).round(1)
    
    # Estimate steps from activity magnitude
    daily['total_steps'] = (daily['magnitude'] * 300).astype(int)
    daily['active_min'] = daily['is_active'].astype(int)
    daily['sedentary_min'] = (1440 - daily['active_min']).clip(300, 900).astype(int)
    
    # Get recording period
    recording_start = data.index.min().strftime('%Y-%m-%dT%H:%M:%SZ')
    recording_end = data.index.max().strftime('%Y-%m-%dT%H:%M:%SZ')
    
    patient = {
        'subject_id': 'PID-20394',
        'name': 'Mr David Chen',
        'dob': '1968-04-15',
        'device': info['Device'],
        'recording_start': recording_start,
        'recording_end': recording_end,
        'mean_daily_steps': int(daily['total_steps'].mean()),
        'total_active_minutes': int(daily['active_min'].sum()),
        'mean_daily_sedentary_minutes': int(daily['sedentary_min'].mean()),
        'mean_heart_rate_bpm': 72,
        'resting_hr_bpm': 62,
        'days_meeting_150min_activity_target': int((daily['active_min'] >= 21).sum()),
        'clinical_context': 'Cardiac rehabilitation post-MI, 6 weeks post-discharge'
    }
    
    return patient


def generate_fhir_bundle(patient):
    """
    Generate FHIR R4 Bundle from patient data using GPT-4o-mini.
    
    Args:
        patient: Patient dictionary with metrics
        
    Returns:
        FHIR Bundle JSON string
    """
    r = client.chat.completions.create(
        model='gpt-4o-mini',
        response_format={'type': 'json_object'},
        messages=[
            {'role': 'system', 'content': FHIR_SYSTEM},
            {'role': 'user', 'content': json.dumps(patient)}
        ],
        temperature=0.0
    )
    return r.choices[0].message.content


def validate_fhir_bundle(fhir_json):
    """
    Validate FHIR Bundle structure.
    
    Args:
        fhir_json: FHIR Bundle JSON string
        
    Returns:
        Tuple of (is_valid, observations_count, error_message)
    """
    try:
        bundle = Bundle.model_validate(json.loads(fhir_json))
        obs_count = 0
        
        if bundle.entry:
            for e in bundle.entry:
                if e.resource and hasattr(e.resource, 'resource_type'):
                    if e.resource.resource_type == 'Observation':
                        obs_count += 1
        
        return True, obs_count, None
    except Exception as e:
        return False, 0, str(e)


def generate_clinical_snapshot(patient):
    """
    Generate clinical snapshot text using GPT-4o-mini.
    
    Args:
        patient: Patient dictionary with metrics
        
    Returns:
        Clinical snapshot text
    """
    eval_prompt = f"""{SNAPSHOT_SYS_FT}\n\nClinical Guidelines to follow:\n{CLINICAL_GUIDELINES}"""
    
    r = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[
            {'role': 'system', 'content': eval_prompt},
            {'role': 'user', 'content': json.dumps(patient)}
        ],
        temperature=0.1
    )
    return r.choices[0].message.content


def generate_html_snapshot(snapshot_text, title="Clinical Snapshot", urgent=False):
    """Generate HTML for clinical snapshot display with improved parsing and styling."""
    color = "#d9534f" if urgent else "#0275d8"
    bg_color = "#fff5f5" if urgent else "#f0f8ff"

    import re

    # Normalize text and insert missing delimiters before section headers split across lines.
    normalized = snapshot_text.replace('\r', ' ')
    normalized = re.sub(
        r'\n\s*(PATIENT|MONITORING PERIOD|KEY FINDINGS(?:\s*\([^)]*\))?|CLINICAL INTERPRETATION|RECOMMENDATION)\s*\|',
        r' | \1 |',
        normalized,
        flags=re.IGNORECASE
    )
    normalized = re.sub(r'\s*\|\s*', ' | ', normalized)
    parts = [part.strip() for part in normalized.split(' | ') if part.strip()]

    sections = {}
    for i in range(0, len(parts), 2):
        key = parts[i].upper()
        value = parts[i + 1] if i + 1 < len(parts) else ''
        if key == 'PATIENT':
            sections['Patient'] = value.strip()
        elif key == 'MONITORING PERIOD':
            sections['Monitoring Period'] = value.strip()
        elif key.startswith('KEY FINDINGS'):
            sections['Key Findings'] = value.strip()
        elif key == 'CLINICAL INTERPRETATION':
            sections['Clinical Interpretation'] = value.strip()
        elif key == 'RECOMMENDATION':
            sections['Recommendation'] = value.strip()

    # Section configuration with icons
    section_config = {
        "Patient": ("👤", "Patient information"),
        "Monitoring Period": ("📅", "Recording timeframe"),
        "Key Findings": ("📊", "Activity, Heart rate, Sedentary metrics"),
        "Clinical Interpretation": ("🔍", "Medical assessment"),
        "Recommendation": ("💡", "Suggested actions")
    }

    html = f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Wearable Clinical Pipeline</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
        <style>
            body {{
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                line-height: 1.6;
            }}
            .container {{
                background-color: white;
                padding: 40px;
                border-radius: 16px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.1);
                margin-bottom: 30px;
            }}
            h1 {{
                color: #1a365d;
                text-align: center;
                margin-bottom: 40px;
                font-size: 2.5em;
                font-weight: 600;
                text-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}

            .snapshot {{
                border: 3px solid {color};
                border-radius: 12px;
                padding: 30px;
                margin: 30px 0;
                background: {bg_color};
                position: relative;
                overflow: hidden;
            }}

            .snapshot::before {{
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 4px;
                background: linear-gradient(90deg, {color}, {color}dd);
            }}

            .snapshot h2 {{
                color: {color};
                margin-top: 0;
                font-size: 1.8em;
                font-weight: 600;
                display: flex;
                align-items: center;
                gap: 10px;
            }}

            .urgent-badge {{
                background: #fed7d7;
                color: #c53030;
                padding: 4px 12px;
                border-radius: 20px;
                font-size: 0.8em;
                font-weight: 600;
                border: 2px solid #feb2b2;
            }}

            .sections {{
                display: grid;
                gap: 20px;
                margin-top: 25px;
            }}

            .section {{
                background: white;
                border-radius: 8px;
                padding: 20px;
                border-left: 4px solid {color};
                box-shadow: 0 2px 8px rgba(0,0,0,0.05);
                transition: transform 0.2s ease, box-shadow 0.2s ease;
            }}

            .section:hover {{
                transform: translateY(-2px);
                box-shadow: 0 4px 16px rgba(0,0,0,0.1);
            }}

            .section-header {{
                display: flex;
                align-items: center;
                gap: 12px;
                margin-bottom: 12px;
                font-weight: 600;
                color: #2d3748;
                font-size: 1.1em;
            }}

            .section-content {{
                color: #4a5568;
                line-height: 1.7;
                font-size: 1em;
            }}

            .metrics-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin-top: 15px;
            }}

            .metric {{
                background: #f7fafc;
                padding: 12px;
                border-radius: 6px;
                border: 1px solid #e2e8f0;
                text-align: center;
            }}

            .metric-value {{
                font-size: 1.4em;
                font-weight: 700;
                color: {color};
                display: block;
            }}

            .metric-label {{
                font-size: 0.9em;
                color: #718096;
                margin-top: 4px;
            }}

            .api-info {{
                background: linear-gradient(135deg, #ebf8ff 0%, #bee3f8 100%);
                padding: 30px;
                border-radius: 12px;
                margin: 30px 0;
                border: 2px solid #90cdf4;
            }}
            .api-info h3 {{
                color: #2b6cb0;
                margin-top: 0;
                font-size: 1.5em;
                display: flex;
                align-items: center;
                gap: 10px;
            }}
            .endpoints {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }}
            .endpoint {{
                background: white;
                padding: 20px;
                border-radius: 8px;
                border-left: 5px solid #3182ce;
                box-shadow: 0 2px 8px rgba(0,0,0,0.05);
                transition: all 0.2s ease;
            }}
            .endpoint:hover {{
                transform: translateY(-2px);
                box-shadow: 0 4px 16px rgba(0,0,0,0.1);
            }}
            .endpoint strong {{
                color: #2b6cb0;
                font-size: 1.1em;
                display: block;
                margin-bottom: 8px;
            }}
            .endpoint p {{
                margin: 0;
                color: #4a5568;
            }}

            .api-info {{
                background: linear-gradient(135deg, #ebf8ff 0%, #bee3f8 100%);
                padding: 30px;
                border-radius: 12px;
                margin: 30px 0;
                border: 2px solid #90cdf4;
            }}
            .api-info h3 {{
                color: #2b6cb0;
                margin-top: 0;
                font-size: 1.5em;
                display: flex;
                align-items: center;
                gap: 10px;
            }}
            .endpoints {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }}
            .endpoint {{
                background: white;
                padding: 20px;
                border-radius: 8px;
                border-left: 5px solid #3182ce;
                box-shadow: 0 2px 8px rgba(0,0,0,0.05);
                transition: all 0.2s ease;
            }}
            .endpoint:hover {{
                transform: translateY(-2px);
                box-shadow: 0 4px 16px rgba(0,0,0,0.1);
            }}
            .endpoint strong {{
                color: #2b6cb0;
                font-size: 1.1em;
                display: block;
                margin-bottom: 8px;
            }}
            .endpoint p {{
                margin: 0;
                color: #4a5568;
            }}

            .footer {{
                text-align: center;
                color: #718096;
                font-size: 0.9em;
                margin-top: 40px;
                padding-top: 20px;
                border-top: 1px solid #e2e8f0;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🏥 Wearable Clinical Pipeline</h1>

            <div class="snapshot">
                <h2>
                    {title}
                    {"<span class='urgent-badge'>⚠️ URGENT</span>" if urgent else ""}
                </h2>

                <div class="sections">
    '''

    # Generate HTML for each section
    for section_name, content in sections.items():
        icon, description = section_config.get(section_name, ("📋", ""))

        # Special handling for Key Findings - extract and highlight metrics
        if section_name == "Key Findings" and any(char.isdigit() for char in content):
            # Try to extract metrics from the content
            metrics_html = ""
            import re

            # Extract step count
            steps_match = re.search(r'(\d+)\s*steps?', content, re.IGNORECASE)
            if steps_match:
                metrics_html += f'<div class="metric"><span class="metric-value">{steps_match.group(1)}</span><span class="metric-label">Daily Steps</span></div>'

            # Extract heart rate
            hr_match = re.search(r'(\d+)\s*bpm', content, re.IGNORECASE)
            if hr_match:
                metrics_html += f'<div class="metric"><span class="metric-value">{hr_match.group(1)}</span><span class="metric-label">Heart Rate (bpm)</span></div>'

            # Extract sedentary time
            sed_match = re.search(r'(\d+)\s*min', content, re.IGNORECASE)
            if sed_match:
                metrics_html += f'<div class="metric"><span class="metric-value">{sed_match.group(1)}</span><span class="metric-label">Sedentary (min)</span></div>'

            if metrics_html:
                content = f'<div class="metrics-grid">{metrics_html}</div><div style="margin-top: 15px; color: #4a5568;">{content}</div>'
            else:
                content = f'<div class="section-content">{content}</div>'
        else:
            content = f'<div class="section-content">{content}</div>'

        html += f'''
                    <div class="section">
                        <div class="section-header">{icon} {section_name}</div>
                        {content}
                    </div>
        '''

    html += '''
                </div>
            </div>
        </div>

        <div class="footer">
            <p>🩺 Clinical AI Pipeline • Powered by GPT-4o-mini • FHIR R4 Compliant</p>
        </div>
    </body>
    </html>
    '''
    return html


@app.route('/', methods=['GET'])
def index():
    """Web interface showing clinical snapshot and API info."""
    # Create default patient data for demonstration
    patient = {
        'subject_id': 'PID-20394',
        'name': 'Mr David Chen',
        'dob': '1968-04-15',
        'device': 'Axivity AX3',
        'recording_start': '2023-06-08T12:21:04Z',
        'recording_end': '2023-06-08T15:19:33Z',
        'mean_daily_steps': 300,
        'total_active_minutes': 15,
        'mean_daily_sedentary_minutes': 900,
        'mean_heart_rate_bpm': 72,
        'resting_hr_bpm': 62,
        'days_meeting_150min_activity_target': 0,
        'clinical_context': 'Cardiac rehabilitation post-MI, 6 weeks post-discharge'
    }
    
    # Generate clinical snapshot
    snapshot_text = generate_clinical_snapshot(patient)
    
    # Generate HTML response
    html_response = generate_html_snapshot(snapshot_text, title="Post-MI Recovery Update", urgent=False)
    
    return html_response


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'ok', 'service': 'Wearable Clinical Pipeline'})


@app.route('/process', methods=['POST'])
def process_wearable_file():
    """
    Process uploaded wearable file and generate clinical snapshot.
    
    Expected multipart/form-data with 'file' field containing .cwa, .gz, or .csv file.
    
    Returns:
        JSON with clinical snapshot, FHIR validation info, and patient metrics
    """
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Verify file extension
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in ['.cwa', '.gz', '.csv']:
            return jsonify({'error': f'Unsupported file format: {file_ext}. Accepted: .cwa, .gz, .csv'}), 400
        
        # Save file to temporary location
        with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name
        
        try:
            # Step 1: Load wearable data
            data, info = load_wearable_data(tmp_path)
            
            # Step 2: Extract patient metrics
            patient = extract_patient_metrics(data, info)
            
            # Step 3: Generate FHIR Bundle
            fhir_bundle = generate_fhir_bundle(patient)
            
            # Step 4: Validate FHIR Bundle
            is_valid, obs_count, fhir_error = validate_fhir_bundle(fhir_bundle)
            
            # Step 5: Generate clinical snapshot
            snapshot = generate_clinical_snapshot(patient)
            
            response = {
                'status': 'success',
                'patient': patient,
                'fhir_validation': {
                    'is_valid': is_valid,
                    'observations_count': obs_count,
                    'error': fhir_error
                },
                'clinical_snapshot': snapshot,
                'recording_duration': {
                    'start': patient['recording_start'],
                    'end': patient['recording_end']
                }
            }
            
            return jsonify(response), 200
            
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error processing file: {error_trace}")
        return jsonify({
            'error': str(e),
            'traceback': error_trace
        }), 500


@app.route('/snapshot', methods=['POST'])
def generate_snapshot_endpoint():
    """
    Generate clinical snapshot from patient data (JSON body).
    
    Expected JSON:
    {
        "subject_id": "PID-123",
        "name": "Patient Name",
        "dob": "1960-01-01",
        "mean_daily_steps": 6000,
        ...
    }
    
    Returns:
        JSON with clinical snapshot text
    """
    try:
        patient = request.get_json()
        
        if not patient:
            return jsonify({'error': 'No patient data provided'}), 400
        
        snapshot = generate_clinical_snapshot(patient)
        
        return jsonify({
            'status': 'success',
            'clinical_snapshot': snapshot
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
