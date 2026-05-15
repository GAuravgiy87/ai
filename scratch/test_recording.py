"""
Test script to verify recording functionality
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import requests
from database.sqlite_manager import SqliteManager

def test_recording_system():
    """Test the recording system"""
    print("=" * 60)
    print("Recording System Diagnostic Test")
    print("=" * 60)
    
    # 1. Check if recordings directory exists
    print("\n1. Checking recordings directory...")
    if os.path.exists("recordings"):
        print("   ✓ recordings/ directory exists")
        # List subdirectories
        for root, dirs, files in os.walk("recordings"):
            level = root.replace("recordings", "").count(os.sep)
            indent = " " * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = " " * 2 * (level + 1)
            for file in files:
                size_mb = os.path.getsize(os.path.join(root, file)) / (1024 * 1024)
                print(f"{subindent}{file} ({size_mb:.2f} MB)")
    else:
        print("   ✗ recordings/ directory does not exist")
        os.makedirs("recordings", exist_ok=True)
        print("   ✓ Created recordings/ directory")
    
    # 2. Check database recordings table
    print("\n2. Checking database recordings...")
    try:
        db = SqliteManager()
        recordings = db.search_recordings()
        print(f"   ✓ Found {len(recordings)} recording entries in database")
        for rec in recordings[:5]:  # Show first 5
            print(f"     - ID: {rec[0]}, Camera: {rec[1]}, Start: {rec[2]}, End: {rec[3]}")
            print(f"       File: {rec[4]}")
            if os.path.exists(rec[4]):
                size_mb = os.path.getsize(rec[4]) / (1024 * 1024)
                print(f"       ✓ File exists ({size_mb:.2f} MB)")
            else:
                print(f"       ✗ File not found")
    except Exception as e:
        print(f"   ✗ Database error: {e}")
    
    # 3. Check camera server status
    print("\n3. Checking camera server...")
    try:
        response = requests.get("http://localhost:9001/health", timeout=2)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✓ Camera server is running")
            print(f"   ✓ Active cameras: {data.get('cameras', [])}")
        else:
            print(f"   ✗ Camera server returned status {response.status_code}")
    except Exception as e:
        print(f"   ✗ Cannot connect to camera server: {e}")
    
    # 4. Check recording settings for each camera
    print("\n4. Checking camera recording settings...")
    try:
        response = requests.get("http://localhost:9001/cameras", timeout=2)
        if response.status_code == 200:
            cameras = response.json()
            for cam in cameras:
                cam_id = cam['id']
                settings_resp = requests.get(f"http://localhost:9001/settings/{cam_id}", timeout=2)
                if settings_resp.status_code == 200:
                    settings = settings_resp.json()
                    enabled = settings.get('recording_enabled', False)
                    recording = settings.get('actually_recording', False)
                    status = "✓" if enabled else "✗"
                    rec_status = "✓" if recording else "✗"
                    print(f"   {status} {cam_id}: Enabled={enabled}, Recording={recording} {rec_status}")
                else:
                    print(f"   ? {cam_id}: Cannot get settings")
    except Exception as e:
        print(f"   ✗ Error checking settings: {e}")
    
    # 5. Check FFmpeg availability
    print("\n5. Checking FFmpeg...")
    try:
        import subprocess
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, timeout=2)
        if result.returncode == 0:
            version_line = result.stdout.decode().split('\n')[0]
            print(f"   ✓ FFmpeg is available: {version_line}")
        else:
            print(f"   ✗ FFmpeg returned error code {result.returncode}")
    except FileNotFoundError:
        print("   ✗ FFmpeg not found in PATH")
    except Exception as e:
        print(f"   ✗ Error checking FFmpeg: {e}")
    
    print("\n" + "=" * 60)
    print("Diagnostic test complete")
    print("=" * 60)

if __name__ == "__main__":
    test_recording_system()
