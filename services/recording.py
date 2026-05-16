"""
RecordingService — FFmpeg-based video recording system.
Based on the VigiLance AI Surveillance System implementation guide.

Architecture:
- One FFmpeg subprocess per active recording
- One daemon writer thread per recording (feeds frames at 10 FPS)
- Frames sourced from camera_results shared dict (rendered frames with overlays)
- Thread-safe state management with locks
"""

import os
import time
import logging
import threading
import subprocess
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class RecordingService:
    """
    Manages per-camera FFmpeg processes and frame writing threads.
    
    State:
        camera_writers: {camera_id: writer_data}
        recording_threads: {camera_id: Thread}
        recording_stop_events: {camera_id: Event}
    
    writer_data structure:
        {
            "process": subprocess.Popen,
            "db_id": int,
            "start_time": float,
            "file_path": str,
            "w": int,
            "h": int
        }
    """
    
    def __init__(self, db_manager, camera_results, results_lock, recordings_dir: str = "./recordings", chunk_duration: int = 3600):
        """
        Initialize the recording service.
        
        Args:
            db_manager: Database manager with start_recording/end_recording methods
            camera_results: Shared dict containing rendered frames {camera_id: {"rendered_frame": np.ndarray}}
            results_lock: threading.Lock protecting camera_results
            recordings_dir: Directory where MP4 files are saved
            chunk_duration: Duration of each recording chunk in seconds (default: 3600 = 1 hour)
        """
        self.db_manager = db_manager
        self.camera_results = camera_results
        self.results_lock = results_lock
        self.recordings_dir = recordings_dir
        self.chunk_duration = chunk_duration
        
        # State dictionaries
        self.camera_writers: Dict[str, Dict[str, Any]] = {}
        self.recording_threads: Dict[str, threading.Thread] = {}
        self.recording_stop_events: Dict[str, threading.Event] = {}
        
        # Single lock protects all three state dicts
        self.writer_lock = threading.Lock()
        
        # Ensure recordings directory exists
        os.makedirs(recordings_dir, exist_ok=True)
        
        logger.info(f"[RecordingService] Initialized with recordings_dir={recordings_dir}, chunk_duration={chunk_duration}s")
    
    def start_recording(self, camera_id: str, w: int, h: int) -> bool:
        """
        Start recording for a camera.
        
        Args:
            camera_id: Camera identifier
            w: Frame width in pixels
            h: Frame height in pixels
        
        Returns:
            True if recording started successfully, False otherwise
        """
        with self.writer_lock:
            if camera_id in self.camera_writers:
                logger.warning(f"[RecordingService] Camera {camera_id} is already recording")
                return False
        
        # Generate timestamped filename with date/camera structure
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        hour_str = now.strftime("%H")
        minute_str = now.strftime("%M")
        second_str = now.strftime("%S")
        
        # Create directory structure: recordings/{date}/{camera_id}/
        camera_dir = os.path.join(self.recordings_dir, date_str, camera_id)
        os.makedirs(camera_dir, exist_ok=True)
        
        # Filename: {hour}_{minute}{second}.mp4 (e.g., 14_3045.mp4 for 2:30:45 PM)
        # This ensures no overwrites - each recording session gets unique filename
        filename = f"{hour_str}_{minute_str}{second_str}.mp4"
        file_path = os.path.join(camera_dir, filename)
        
        # Build FFmpeg command
        ffmpeg_cmd = [
            "ffmpeg",
            "-f", "rawvideo",              # Input format: raw pixel data
            "-vcodec", "rawvideo",         # No input encoding
            "-s", f"{w}x{h}",             # Frame dimensions MUST match actual frame size
            "-pix_fmt", "bgr24",           # OpenCV default pixel order (not RGB!)
            "-r", "10",                    # Input frame rate = target frame rate (10 FPS)
            "-i", "-",                     # Read from stdin
            "-vcodec", "libx264",          # H.264 encoding (widely compatible)
            "-pix_fmt", "yuv420p",         # Required for browser/player compatibility
            "-preset", "ultrafast",        # Minimize CPU usage
            "-crf", "28",                  # Quality factor: 18=high quality, 28=smaller file
            "-force_key_frames", "expr:gte(t,n_forced*2)",  # Keyframe every 2s; improves seek and partial-file recovery
            "-movflags", "+faststart",     # Write moov atom at start; makes file playable even if not finalized cleanly
            file_path                      # Output file path
        ]
        
        try:
            # Start FFmpeg subprocess
            process = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                bufsize=10**8  # Large buffer for stdin
            )
            
            # Give FFmpeg a moment to start
            time.sleep(0.1)
            if process.poll() is not None:
                logger.error(f"[RecordingService] FFmpeg failed to start for {camera_id}")
                return False
            
            # Register in database
            db_id = self.db_manager.start_recording(camera_id, file_path)
            if db_id is None:
                logger.error(f"[RecordingService] Failed to create DB entry for {camera_id}")
                process.kill()
                return False
            
            logger.info(f"[RecordingService] Database entry created: ID={db_id}")
            
            # Create stop event
            stop_event = threading.Event()
            
            # Store writer data BEFORE starting thread (prevents race condition)
            with self.writer_lock:
                self.camera_writers[camera_id] = {
                    "process": process,
                    "db_id": db_id,
                    "start_time": time.time(),
                    "file_path": file_path,
                    "w": w,
                    "h": h
                }
                self.recording_stop_events[camera_id] = stop_event
            
            # Start writer thread
            writer_thread = threading.Thread(
                target=self._writer_loop,
                args=(camera_id, stop_event),
                daemon=True,
                name=f"RecWriter-{camera_id}"
            )
            writer_thread.start()
            
            # Store thread reference
            with self.writer_lock:
                self.recording_threads[camera_id] = writer_thread
            
            # Consume FFmpeg stderr in background
            stderr_thread = threading.Thread(
                target=self._log_ffmpeg_stderr,
                args=(process.stderr, camera_id),
                daemon=True,
                name=f"FFmpegLog-{camera_id}"
            )
            stderr_thread.start()
            
            logger.info(f"[RecordingService] Successfully started recording for {camera_id} -> {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"[RecordingService] Failed to start recording for {camera_id}: {e}", exc_info=True)
            return False
    
    def _finalize_recording(self, camera_id: str):
        """
        Finalize the current recording chunk (close FFmpeg, update DB).
        Called by writer thread when rotation is needed or recording stops.
        
        Args:
            camera_id: Camera identifier
        """
        with self.writer_lock:
            writer_data = self.camera_writers.get(camera_id)
            if not writer_data:
                return
        
        process = writer_data.get("process")
        db_id = writer_data.get("db_id")
        file_path = writer_data.get("file_path")
        
        # Close FFmpeg gracefully by closing stdin, which signals EOF and triggers moov atom write
        if process:
            try:
                if process.stdin:
                    try:
                        process.stdin.flush()
                    except Exception:
                        pass
                    try:
                        process.stdin.close()
                    except Exception:
                        pass
                process.wait(timeout=15)
                logger.info(f"[RecordingService] FFmpeg finalized for {camera_id}")
            except subprocess.TimeoutExpired:
                logger.warning(f"[RecordingService] FFmpeg timeout for {camera_id}, killing")
                process.kill()
                process.wait()
            except Exception as e:
                logger.error(f"[RecordingService] Error finalizing FFmpeg for {camera_id}: {e}")
                if process:
                    process.kill()
        
        # Update database
        if db_id:
            self.db_manager.end_recording(db_id)
            logger.info(f"[RecordingService] Database updated for {camera_id}, ID={db_id}")
        
        # Verify file
        if file_path and os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            logger.info(f"[RecordingService] Recording saved: {file_path} ({file_size / (1024*1024):.2f} MB)")
        else:
            logger.warning(f"[RecordingService] Recording file not found: {file_path}")
    
    def stop_recording(self, camera_id: str) -> bool:
        """
        Stop recording for a camera.
        
        Args:
            camera_id: Camera identifier
        
        Returns:
            True if recording stopped successfully, False otherwise
        """
        # Pop writer data and stop event from dicts (atomic operation)
        with self.writer_lock:
            writer_data = self.camera_writers.pop(camera_id, None)
            stop_event = self.recording_stop_events.pop(camera_id, None)
            writer_thread = self.recording_threads.pop(camera_id, None)
        
        if writer_data is None:
            logger.warning(f"[RecordingService] No active recording for {camera_id}")
            return False
        
        # Signal writer thread to stop
        if stop_event:
            stop_event.set()
        
        # Wait for writer thread to exit
        if writer_thread:
            writer_thread.join(timeout=5)
            if writer_thread.is_alive():
                logger.warning(f"[RecordingService] Writer thread did not exit cleanly for {camera_id}")
        
        # Finalization is handled by _finalize_recording() called from writer thread
        return True
    
    def _writer_loop(self, camera_id: str, stop_event: threading.Event):
        """
        Writer thread: feeds frames to FFmpeg stdin at 10 FPS.
        Automatically rotates recording every chunk_duration seconds.
        
        Args:
            camera_id: Camera identifier
            stop_event: Event to signal thread to stop
        """
        logger.info(f"[RecordingService] Writer thread started for {camera_id}")
        frame_count = 0
        
        while not stop_event.is_set():
            try:
                # Get writer data
                with self.writer_lock:
                    if camera_id not in self.camera_writers:
                        logger.info(f"[RecordingService] Camera {camera_id} not in writers, stopping thread")
                        break
                    writer_data = self.camera_writers[camera_id]
                    process = writer_data.get("process")
                    start_time = writer_data.get("start_time")
                
                # Check if we need to rotate (hourly)
                current_time = time.time()
                recording_duration = current_time - start_time
                
                if recording_duration >= self.chunk_duration:
                    logger.info(f"[RecordingService] Hourly rotation for {camera_id} (duration: {recording_duration:.0f}s)")
                    
                    # Finalize current recording
                    self._finalize_recording(camera_id)
                    
                    # Get frame dimensions for new recording
                    with self.results_lock:
                        frame_data = self.camera_results.get(camera_id, {})
                        frame = frame_data.get("rendered_frame")
                    
                    if frame is not None:
                        h, w = frame.shape[:2]
                        logger.info(f"[RecordingService] Starting new recording chunk for {camera_id}")
                        # Start new recording (this will update camera_writers with new process)
                        if self.start_recording(camera_id, w, h):
                            # Recording restarted successfully, this thread can exit
                            logger.info(f"[RecordingService] Rotation complete for {camera_id}, thread exiting")
                            return
                        else:
                            logger.error(f"[RecordingService] Failed to start new recording after rotation for {camera_id}")
                            break
                    else:
                        logger.warning(f"[RecordingService] No frame available for rotation, stopping recording for {camera_id}")
                        break
                
                # Get latest rendered frame
                with self.results_lock:
                    if camera_id in self.camera_results:
                        frame = self.camera_results[camera_id].get("rendered_frame")
                    else:
                        frame = None
                
                # Write frame to FFmpeg
                if frame is not None and process and process.poll() is None:
                    try:
                        process.stdin.write(frame.tobytes())
                        frame_count += 1
                        
                        # Log progress every 600 frames (~60 seconds)
                        if frame_count % 600 == 0:
                            logger.info(f"[RecordingService] {camera_id}: {frame_count} frames written ({recording_duration/60:.1f} min)")
                    
                    except (IOError, BrokenPipeError) as e:
                        logger.error(f"[RecordingService] Pipe error for {camera_id}: {e}")
                        break
                    except Exception as e:
                        logger.error(f"[RecordingService] Write error for {camera_id}: {e}")
                        break
                
                elif process and process.poll() is not None:
                    logger.warning(f"[RecordingService] FFmpeg process died for {camera_id}")
                    break
                
                # Sleep for 10 FPS (0.1 seconds between frames)
                time.sleep(0.1)
            
            except Exception as e:
                logger.error(f"[RecordingService] Thread error for {camera_id}: {e}")
                time.sleep(1)
        
        logger.info(f"[RecordingService] Writer thread stopped for {camera_id}, wrote {frame_count} frames")
        
        # Only finalize if we haven't already (rotation path already finalized)
        with self.writer_lock:
            if camera_id in self.camera_writers:
                # Finalize current recording (crash or stop, not rotation)
                self._finalize_recording(camera_id)
    
    def _log_ffmpeg_stderr(self, stderr_pipe, camera_id: str):
        """
        Background thread to consume FFmpeg stderr output.
        
        Args:
            stderr_pipe: FFmpeg stderr pipe
            camera_id: Camera identifier
        """
        try:
            for line in iter(stderr_pipe.readline, b''):
                msg = line.decode().strip()
                if msg:
                    if "error" in msg.lower():
                        logger.error(f"[FFmpeg:{camera_id}] {msg}")
                    else:
                        logger.debug(f"[FFmpeg:{camera_id}] {msg}")
        except Exception as e:
            logger.error(f"[FFmpeg:{camera_id}] Error reading stderr: {e}")
        finally:
            stderr_pipe.close()
    
    def is_recording(self, camera_id: str) -> bool:
        """
        Check if a camera is currently recording.
        
        Args:
            camera_id: Camera identifier
        
        Returns:
            True if recording, False otherwise
        """
        with self.writer_lock:
            return camera_id in self.camera_writers
    
    def cleanup_all(self):
        """Stop all active recordings. Called on system shutdown."""
        with self.writer_lock:
            camera_ids = list(self.camera_writers.keys())
        
        if not camera_ids:
            logger.info("[RecordingService] No active recordings to cleanup")
            return
        
        logger.info(f"[RecordingService] Cleaning up {len(camera_ids)} active recording(s)...")
        for camera_id in camera_ids:
            try:
                self.stop_recording(camera_id)
            except Exception as e:
                logger.error(f"[RecordingService] Error stopping recording for {camera_id}: {e}")
    
    def start_management_loop(self):
        """
        Start the management loop that monitors recordings and handles automatic rotation.
        This should be called once after initialization.
        """
        management_thread = threading.Thread(
            target=self._management_loop,
            daemon=True,
            name="RecordingManagement"
        )
        management_thread.start()
        logger.info("[RecordingService] Management loop started")
    
    def _management_loop(self):
        """
        Management loop that:
        1. Monitors active recordings for hourly rotation
        2. Restarts recordings after rotation
        3. Handles crash recovery
        4. Auto-starts recordings for cameras without them
        """
        logger.info("[RecordingService] Management loop running")
        
        while True:
            try:
                time.sleep(10)  # Check every 10 seconds
                
                # Get list of cameras that should be recording
                with self.writer_lock:
                    active_cameras = list(self.camera_writers.keys())
                    dead_threads = []
                    
                    # Check for dead writer threads (crash recovery)
                    for camera_id in active_cameras:
                        thread = self.recording_threads.get(camera_id)
                        if thread and not thread.is_alive():
                            dead_threads.append(camera_id)
                
                # Restart recordings for dead threads (crash recovery)
                for camera_id in dead_threads:
                    logger.warning(f"[RecordingService] Detected dead writer thread for {camera_id}, restarting...")
                    
                    # Clean up the dead recording
                    with self.writer_lock:
                        self.camera_writers.pop(camera_id, None)
                        self.recording_threads.pop(camera_id, None)
                        self.recording_stop_events.pop(camera_id, None)
                    
                    # Get frame dimensions and restart
                    with self.results_lock:
                        frame_data = self.camera_results.get(camera_id, {})
                        frame = frame_data.get("rendered_frame")
                    
                    if frame is not None:
                        h, w = frame.shape[:2]
                        logger.info(f"[RecordingService] Restarting recording for {camera_id}")
                        self.start_recording(camera_id, w, h)
                    else:
                        logger.warning(f"[RecordingService] Cannot restart {camera_id}, no frame available")
                
                # Auto-start recordings for cameras that have frames but no recording
                with self.results_lock:
                    cameras_with_frames = list(self.camera_results.keys())
                
                for camera_id in cameras_with_frames:
                    with self.writer_lock:
                        is_recording = camera_id in self.camera_writers
                    
                    if not is_recording:
                        # Try to start recording
                        with self.results_lock:
                            frame_data = self.camera_results.get(camera_id, {})
                            frame = frame_data.get("rendered_frame")
                        
                        if frame is not None:
                            h, w = frame.shape[:2]
                            logger.info(f"[RecordingService] Auto-starting recording for {camera_id}")
                            self.start_recording(camera_id, w, h)
                
            except Exception as e:
                logger.error(f"[RecordingService] Management loop error: {e}", exc_info=True)
                time.sleep(5)
