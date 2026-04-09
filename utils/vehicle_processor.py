import cv2
import numpy as np
import os
import torch
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class VehicleProcessor:
    def __init__(self, device='cpu'):
        self.device = device
        self.reader = None
        self.initialized = False
        
    def _lazy_init(self):
        """Initialize EasyOCR and Specialized models only when first vehicle is detected to save VRAM."""
        if self.initialized:
            return
        
        try:
            import easyocr
            import torch
            
            # Simplified GPU check for EasyOCR
            use_gpu = False
            if torch.cuda.is_available():
                use_gpu = True
            elif 'GPU' in str(self.device).upper():
                use_gpu = True
                
            self.reader = easyocr.Reader(['en'], gpu=use_gpu)
            logger.info(f"[VehicleProcessor] EasyOCR initialized (GPU: {use_gpu}) on {self.device}")
            self.initialized = True
        except Exception as e:
            logger.error(f"[VehicleProcessor] Failed to init EasyOCR: {e}")

    def process_vehicle(self, frame, bbox, v_type):
        """
        Handle a detected vehicle: ALPR, Occupancy, Helmet.
        bbox: [x, y, w, h]
        """
        self._lazy_init()
        
        x, y, w, h = [int(v) for v in bbox]
        # Pad slightly
        x1, y1 = max(0, x - 10), max(0, y - 10)
        x2, y2 = min(frame.shape[1], x + w + 10), min(frame.shape[0], y + h + 10)
        v_crop = frame[y1:y2, x1:x2]
        
        if v_crop.size == 0:
            return None
            
        result = {
            'type': v_type,
            'plate_text': 'Pending',
            'plate_crop': None,
            'person_count': 0,
            'helmets_on': False,
            'metadata': {}
        }
        
        # 1. ALPR (License Plate)
        if v_type in ['car', 'motorcycle', 'bus', 'truck']:
            plate_text, plate_crop = self._extract_plate(v_crop)
            result['plate_text'] = plate_text
            result['plate_crop'] = plate_crop
            
        # 2. Safety Compliance (Two-wheelers)
        if v_type == 'motorcycle':
             # This will be handled in the main loop by checking overlapping person boxes
             pass
             
        return result

    def _extract_plate(self, v_crop):
        """Find and read license plate in a vehicle crop."""
        if not self.reader:
            return "OCR Error", None
            
        try:
            # EasyOCR can directly find text regions
            # License plates are usually in the lower 60% of a vehicle crop
            h, w = v_crop.shape[:2]
            search_region = v_crop[int(h*0.3):, :] # Look in bottom 70%
            
            # Run OCR
            results = self.reader.readtext(search_region)
            
            # Simple heuristic: find text that looks like a number plate (regex-like)
            # India format example: MH 12 AB 1234
            plates = []
            for (bbox, text, prob) in results:
                # Clean text: remove spaces and symbols
                clean = "".join(c for c in text if c.isalnum()).upper()
                if 4 <= len(clean) <= 12: # Reasonable length for a plate
                    plates.append((clean, prob, bbox))
            
            if not plates:
                return "Unknown", None
                
            # Pick highest confidence plate
            plates.sort(key=lambda x: x[1], reverse=True)
            best_text, prob, plate_bbox = plates[0]
            
            # Crop the plate from search_region
            (tl, tr, br, bl) = plate_bbox
            px1, py1 = int(tl[0]), int(tl[1])
            px2, py2 = int(br[0]), int(br[1])
            plate_crop = search_region[max(0, py1-5):min(h, py2+5), max(0, px1-5):min(w, px2+5)]
            
            return best_text, plate_crop
            
        except Exception as e:
            logger.error(f"[VehicleProcessor] Plate extraction error: {e}")
            return "Error", None
