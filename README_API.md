# Wearable Clinical Pipeline - Backend API

Flask-based REST API for processing wearable device data (.cwa, .gz, .csv) and generating clinical snapshots with FHIR R4 validation.

## Features

- **File Upload**: Accept .cwa (Axivity), .gz (compressed), and .csv wearable data files
- **Data Processing**: Extract activity metrics from accelerometer data
- **FHIR R4 Bundle Generation**: Create validated FHIR bundles with Observation resources
- **Clinical Snapshots**: Generate clinical summaries using GPT-4o-mini
- **Clinical Guardrails**: Apply evidence-based clinical guidelines for cardiac rehabilitation

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Java Environment (macOS)

The app uses `actipy` which requires Java to read .cwa files:

```bash
brew install openjdk
```

The app automatically sets `JAVA_HOME` to `/usr/local/opt/openjdk` on startup.

### 3. Configure Environment Variables

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your-api-key-here
```

## Running the App

### Start the Flask Server

```bash
python app.py
```

The API will be available at `http://localhost:5000`

### Test API Health

```bash
curl http://localhost:5000/health
```

Expected response:
```json
{
  "status": "ok",
  "service": "Wearable Clinical Pipeline"
}
```

## API Endpoints

### 1. POST `/process`

Upload and process a wearable data file.

**Request:**
```bash
curl -X POST -F "file=@tiny-sample.cwa" http://localhost:5000/process
```

**Request:** multipart/form-data
- `file`: Wearable data file (.cwa, .gz, or .csv)

**Response (200 OK):**
```json
{
  "status": "success",
  "patient": {
    "subject_id": "PID-20394",
    "name": "Mr David Chen",
    "dob": "1968-04-15",
    "device": "Axivity",
    "recording_start": "2023-06-08T12:21:04Z",
    "recording_end": "2023-06-08T12:40:45Z",
    "mean_daily_steps": 5320,
    "total_active_minutes": 145,
    "mean_daily_sedentary_minutes": 520,
    "mean_heart_rate_bpm": 72,
    "resting_hr_bpm": 62,
    "days_meeting_150min_activity_target": 5
  },
  "fhir_validation": {
    "is_valid": true,
    "observations_count": 4,
    "error": null
  },
  "clinical_snapshot": "PATIENT: Mr David Chen...\nMONITORING PERIOD: ...\nKEY FINDINGS: ...",
  "recording_duration": {
    "start": "2023-06-08T12:21:04Z",
    "end": "2023-06-08T12:40:45Z"
  }
}
```

**Response (400 Bad Request):**
```json
{
  "error": "Unsupported file format: .txt"
}
```

**Response (500 Server Error):**
```json
{
  "error": "Java runtime not found",
  "traceback": "..."
}
```

---

### 2. POST `/snapshot`

Generate a clinical snapshot from patient data (JSON).

**Request:**
```bash
curl -X POST http://localhost:5000/snapshot \
  -H "Content-Type: application/json" \
  -d '{
    "subject_id": "PID-123",
    "name": "Jane Doe",
    "dob": "1965-05-20",
    "mean_daily_steps": 7200,
    "total_active_minutes": 280,
    "mean_daily_sedentary_minutes": 420,
    "mean_heart_rate_bpm": 71,
    "resting_hr_bpm": 58
  }'
```

**Response (200 OK):**
```json
{
  "status": "success",
  "clinical_snapshot": "PATIENT: Jane Doe...\nMONITORING PERIOD: ..."
}
```

---

### 3. GET `/health`

Health check endpoint.

**Request:**
```bash
curl http://localhost:5000/health
```

**Response:**
```json
{
  "status": "ok",
  "service": "Wearable Clinical Pipeline"
}
```

## Example Usage

### Using Python Client

```bash
python example_client.py tiny-sample.cwa
```

This will:
1. Connect to the local Flask API
2. Upload the file
3. Process and generate clinical snapshot
4. Display results

### Using cURL

```bash
# Process a .cwa file
curl -X POST -F "file=@tiny-sample.cwa" http://localhost:5000/process | python -m json.tool

# Generate snapshot from JSON
curl -X POST http://localhost:5000/snapshot \
  -H "Content-Type: application/json" \
  -d @patient_data.json | python -m json.tool
```

## Data Processing Pipeline

1. **Load Wearable Data**: Read .cwa, .gz, or .csv file using actipy
2. **Extract Metrics**:
   - Calculate acceleration magnitude from tri-axial accelerometer (x, y, z)
   - Detect active vs. sedentary periods
   - Resample to 1-minute intervals
   - Generate daily summaries
3. **Generate FHIR Bundle**: Use GPT-4o-mini to create FHIR R4 transactions with:
   - Step count (LOINC: 41950-7)
   - Physical activity active minutes (LOINC: 55423-8)
   - Heart rate (LOINC: 8867-4)
   - Sedentary time (LOINC: 82291-6)
4. **Validate FHIR**: Verify Bundle structure and count Observations
5. **Generate Clinical Snapshot**: Use GPT-4o-mini with clinical guidelines to create readable summary
6. **Apply Guardrails**:
   - RHR > 80 bpm triggers review
   - Steps < 5000 + sedentary > 540 min → physiotherapist intervention
   - HR > 120 bpm or 50% activity drop → urgent alert

## Clinical Snapshot Format

```
PATIENT: [Name], DOB [Date]. [Clinical Context].
MONITORING PERIOD: [Duration] continuous wearable monitoring.
KEY FINDINGS:
- Activity: [Steps/day], [Active minutes/week].
- Heart rate: Mean HR [bpm], resting HR [bpm].
- Sedentary behaviour: [Minutes/day].
CLINICAL INTERPRETATION: [Assessment of findings].
RECOMMENDATION: [Clinical action plan].
```

## Supported File Formats

| Format | Extension | Tool |
|--------|-----------|------|
| Axivity | .cwa | actipy.read_device() |
| Compressed | .gz | actipy.read_device() |
| CSV | .csv | pandas.read_csv() |

CSV files should include columns: `time`, `x`, `y`, `z`, `temperature`, `light`

## Environment Requirements

- **Python**: 3.8+
- **OpenAI API Key**: Required for GPT-4o-mini integration
- **Java**: Required for .cwa file reading (install via `brew install openjdk`)

## Troubleshooting

### Java Not Found Error

```
Error: The operation couldn't be completed. Unable to locate a Java Runtime.
```

**Solution:**
```bash
brew install openjdk
```

### API Connection Failed

```
Error: Could not connect to http://localhost:5000
```

**Solution:** Ensure Flask app is running in another terminal:
```bash
python app.py
```

### OpenAI API Error

```
Error: OpenAI API key not found
```

**Solution:** Create `.env` file with:
```
OPENAI_API_KEY=sk-...
```

## Performance Notes

- First request to `/process` takes ~30-60 seconds (includes FHIR generation and clinical snapshot generation)
- Subsequent requests benefit from OpenAI API caching
- File upload time depends on file size (typically < 5 seconds for .cwa files)

## License

MIT
