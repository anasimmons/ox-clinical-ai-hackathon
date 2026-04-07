"""
Example client for the Wearable Clinical Pipeline API

Usage:
    python example_client.py <path_to_wearable_file>

Supported file formats: .cwa, .gz, .csv
"""

import requests
import json
import sys
from pathlib import Path


def upload_and_process(file_path, api_url='http://localhost:5001'):
    """
    Upload a wearable file to the API and get clinical snapshot.
    
    Args:
        file_path: Path to .cwa, .gz, or .csv file
        api_url: Base URL of the Flask API (default: localhost:5001)
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        return
    
    if file_path.suffix.lower() not in ['.cwa', '.gz', '.csv']:
        print(f"Error: Unsupported file format: {file_path.suffix}")
        print("Supported formats: .cwa, .gz, .csv")
        return
    
    print(f"Uploading {file_path.name}...")
    
    with open(file_path, 'rb') as f:
        files = {'file': f}
        try:
            response = requests.post(
                f'{api_url}/process',
                files=files,
                timeout=300  # 5 minute timeout for processing
            )
        except requests.exceptions.ConnectionError:
            print(f"Error: Could not connect to {api_url}")
            print("Make sure the Flask app is running: python app.py")
            return
        except requests.exceptions.Timeout:
            print("Error: Request timed out. Processing took too long.")
            return
    
    if response.status_code == 200:
        result = response.json()
        
        print("\n" + "="*80)
        print("CLINICAL SNAPSHOT")
        print("="*80 + "\n")
        print(result['clinical_snapshot'])
        print("\n" + "="*80)
        
        print("\nPATIENT METRICS:")
        patient = result['patient']
        print(f"  Name: {patient['name']}")
        print(f"  DOB: {patient['dob']}")
        print(f"  Mean Daily Steps: {patient['mean_daily_steps']}")
        print(f"  Total Active Minutes: {patient['total_active_minutes']}")
        print(f"  Mean Sedentary Minutes: {patient['mean_daily_sedentary_minutes']}")
        print(f"  Recording Period: {patient['recording_start']} to {patient['recording_end']}")
        
        print("\nFHIR VALIDATION:")
        fhir = result['fhir_validation']
        print(f"  Valid: {fhir['is_valid']}")
        print(f"  Observations: {fhir['observations_count']}")
        if fhir['error']:
            print(f"  Error: {fhir['error']}")
    else:
        print(f"Error: {response.status_code}")
        print(response.json())


def test_api_health(api_url='http://localhost:5000'):
    """Test if the API is healthy."""
    try:
        response = requests.get(f'{api_url}/health', timeout=5)
        if response.status_code == 200:
            print(f"✓ API is healthy: {response.json()}")
            return True
        else:
            print(f"✗ API returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"✗ Could not connect to {api_url}")
        return False


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python example_client.py <path_to_wearable_file>")
        print("\nExample: python example_client.py tiny-sample.cwa")
        print("\nFirst, start the Flask app in another terminal:")
        print("  python app.py  # Runs on port 5001")
        sys.exit(1)
    
    file_path = sys.argv[1]
    api_url = sys.argv[2] if len(sys.argv) > 2 else 'http://localhost:5001'
    
    print(f"Testing connection to {api_url}...")
    if test_api_health(api_url):
        print("\nProcessing file...")
        upload_and_process(file_path, api_url)
    else:
        print("API is not available. Start the Flask app first: python app.py")
