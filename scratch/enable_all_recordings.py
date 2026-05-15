"""
Enable automatic recording for all existing cameras
Run this once to enable recording for cameras that were added before the fix
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.sqlite_manager import SqliteManager

def enable_all_recordings():
    """Enable recording for all cameras in the database"""
    print("=" * 60)
    print("Enabling Automatic Recording for All Cameras")
    print("=" * 60)
    
    try:
        db = SqliteManager()
        cameras = db.get_cameras()
        
        if not cameras:
            print("\n✗ No cameras found in database")
            return
        
        print(f"\nFound {len(cameras)} camera(s):")
        
        for cam_id, source in cameras:
            print(f"\n  Camera: {cam_id}")
            print(f"  Source: {source}")
            
            # Enable recording
            db.set_camera_recording(cam_id, True)
            
            # Verify
            enabled = bool(db.get_camera_recording_setting(cam_id))
            if enabled:
                print(f"  ✓ Recording enabled")
            else:
                print(f"  ✗ Failed to enable recording")
        
        print("\n" + "=" * 60)
        print("Done! All cameras now have automatic recording enabled.")
        print("Restart the application for changes to take effect.")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    enable_all_recordings()
