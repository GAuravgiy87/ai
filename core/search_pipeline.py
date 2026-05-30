import cv2
import numpy as np
import os
import logging
from typing import List

logger = logging.getLogger(__name__)

# We need access to the initialized models for scanning.
# To avoid circular imports, we will provide a module-level setup function.
_search_recognizer = None
_search_detector = None

def init_search_models(detector, recognizer):
    global _search_detector, _search_recognizer
    _search_detector = detector
    _search_recognizer = recognizer

def scan_video_for_person(video_path: str, target_encoding: np.ndarray, sample_interval: int = 15) -> List[dict]:
    """
    Optimized high-speed video search using GPU:
    1. YOLOv8 (GPU) detects persons first (fast skip for empty frames).
    2. Crops persons and uses Batch Face Recognition (GPU).
    3. Results are aggregated into segments.
    """
    if not _search_recognizer or not _search_detector:
        logger.warning("[Pipeline] Video scan requested but models are not initialized.")
        return []

    res = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return res

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    f_cnt = 0
    c_seg = None
    l_m_f = -1
    g_gap = int(fps * 3)  # 3-second gap for segmenting
    
    # Batch settings — increased to 32 for massive GPU speedup in forensic scan
    BATCH_SIZE = 32
    pending_batch_frames = []
    pending_batch_indices = []

    logger.info(f"[Search] Starting GPU-accelerated scan on {os.path.basename(video_path)} ({total_frames} frames)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if f_cnt % sample_interval == 0:
            # Step 1: Fast YOLO person detection first (GPU)
            # This is much faster than running face detection on every empty frame
            dets = _search_detector.detect(frame)
            if dets:
                # We found people! Collect person crops for batch recognition
                person_boxes = [d[0] for d in dets]
                # To keep it simple but fast, we take the most prominent person if multiple
                # or we could batch ALL persons. Let's batch ALL persons in this frame.
                
                # However, to avoid exploding the batch, we limit to 1 per frame for search
                # and add this frame to the batch.
                bx, by, bw, bh = person_boxes[0]
                # Expand box slightly for face detection
                pad_w, pad_h = bw * 0.1, bh * 0.1
                face_box = [bx - pad_w, by - pad_h, bx + bw + pad_w, by + bh * 0.5 + pad_h]
                
                pending_batch_frames.append((frame.copy(), face_box))
                pending_batch_indices.append(f_cnt)

            # Step 2: Process batch if full
            if len(pending_batch_frames) >= BATCH_SIZE:
                # BUG-06 fix: _process_search_batch manages state via res_list directly
                _process_search_batch(pending_batch_frames, pending_batch_indices,
                                      target_encoding, fps, g_gap, res)
                pending_batch_frames.clear()
                pending_batch_indices.clear()

        f_cnt += 1

    # Process final partial batch
    if pending_batch_frames:
        _process_search_batch(pending_batch_frames, pending_batch_indices,
                              target_encoding, fps, g_gap, res)

    cap.release()
    logger.info(f"[Search] Scan complete. Found {len(res)} segments.")
    return res

def _process_search_batch(batch, indices, target_encoding, fps, g_gap, res_list):
    """
    Run TRUE batch recognition across multiple frames and update results list.
    SPEED: Now uses recognize_multi_frame_batch for 4x+ performance gain.
    """
    # Run entire batch in one GPU call
    batch_results = _search_recognizer.recognize_multi_frame_batch(batch)
    
    # Target encoding should be normalized for comparison
    target_v = target_encoding / np.linalg.norm(target_encoding)

    for i, (name, conf, enc) in enumerate(batch_results):
        f_idx = indices[i]
        if enc is None: continue

        match_found = False
        match_conf = 0.0

        # High-accuracy normalized L2 comparison
        dist = float(np.linalg.norm(target_v - enc))
        # 1.05 is the sweet spot for Forensic search accuracy
        if dist < 1.05:
            match_found = True
            match_conf = max(0.0, 1.0 - (dist / 1.15))

        if match_found:
            sec  = f_idx / fps
            tstr = f"{int(sec//60)}:{int(sec%60):02d}"

            # Extend the last segment if within gap, otherwise start a new one
            if res_list and (f_idx - res_list[-1]["end_frame"]) <= g_gap:
                res_list[-1]["end_seconds"]   = sec
                res_list[-1]["end_timestamp"] = tstr
                res_list[-1]["end_frame"]     = f_idx
                res_list[-1]["confidence"]    = max(res_list[-1]["confidence"], match_conf)
            else:
                res_list.append({
                    "start_seconds": sec, "start_timestamp": tstr,
                    "end_seconds":   sec, "end_timestamp":   tstr,
                    "confidence":    match_conf,
                    "start_frame":   f_idx, "end_frame": f_idx,
                })
