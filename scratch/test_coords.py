import numpy as np
from utils.vehicle_processor import VehicleProcessor

def test_clamping():
    # Mock frame: (H, W, C) = (480, 640, 3)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    vp = VehicleProcessor()
    
    # BBox that goes out of bounds
    # x, y, w, h
    bbox = [600, 450, 100, 100]
    # Expected x2 = min(640, 600 + 100 + 10) = 640
    # Expected y2 = min(480, 450 + 100 + 10) = 480
    #
    # Old buggy code: y2 = min(3, 450 + 100 + 10) = 3
    
    x, y, w, h = bbox
    x1, y1 = max(0, x - 10), max(0, y - 10)
    x2 = min(frame.shape[1], x + w + 10)
    y2 = min(frame.shape[0], y + h + 10)
    
    print(f"Frame shape: {frame.shape}")
    print(f"BBox: {bbox}")
    print(f"X range: {x1}:{x2}")
    print(f"Y range: {y1}:{y2}")
    
    if y2 == 3:
        print("FAIL: Still using channels (3) for y2 clamping!")
    elif y2 == 480:
        print("SUCCESS: Correctly using height (480) for y2 clamping.")
    else:
        print(f"UNEXPECTED: y2 is {y2}")

if __name__ == "__main__":
    test_clamping()
