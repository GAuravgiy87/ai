import sys
import os
import numpy as np

# Mock the parts of tracker for testing
sys.path.append(os.getcwd())
try:
    from utils.tracker import ObjectTracker
except ImportError:
    print("Could not import ObjectTracker. Please run from project root.")
    sys.exit(1)

def simulate():
    print("--- STARTING TRACKER SIMULATION (Zero-Ghosting Test) ---")
    tracker = ObjectTracker(max_age=10, n_init=1, iou_threshold=0.15)
    
    # Frame 1: Person A detected
    print("\nFrame 1: Person detected at [100, 100, 50, 50]")
    dets1 = [([100, 100, 50, 50], 0.9, 'person')]
    active1 = tracker.update(dets1)
    print(f"Active Tracks: {active1}")
    
    # Frame 2: Person A still there
    print("\nFrame 2: Person still there (slightly shifted)")
    dets2 = [([105, 105, 50, 50], 0.9, 'person')]
    active2 = tracker.update(dets2)
    print(f"Active Tracks: {active2}")
    
    # Frame 3: Person A LEAVES (Empty detections)
    print("\nFrame 3: Person LEAVES (NO detections fed)")
    active3 = tracker.update([])
    print(f"Active Tracks: {active3}")
    
    if len(active3) == 0:
        print("\nSUCCESS: Ghosting resolved! Box disappeared instantly.")
    else:
        print("\nFAILURE: Ghosting detected! Box still stuck.")

    # Frame 4: Person Re-appears in same spot later
    print("\nFrame 4: Person re-appears after 1 frame gap")
    active4 = tracker.update([]) # frame 4 empty
    dets5 = [([110, 110, 50, 50], 0.9, 'person')]
    active5 = tracker.update(dets5) # frame 5 detection
    print(f"Frame 5 Active Tracks: {active5}")
    
    # Check if ID persisted (if IoU matched successfully)
    if active1[0]['id'] == active5[0]['id']:
        print("SUCCESS: ID Persisted correctly.")
    else:
        print("NOTE: ID Reset (New person or age exceeded).")

if __name__ == "__main__":
    simulate()
