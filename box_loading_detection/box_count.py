"""
BOX LOADING DETECTION AND COUNTING SYSTEM
==========================================
Detects boxes using YOLO, tracks them with ByteTrack, and counts when they cross a line.

FLOW: Video Frame → Detect → Track → Check Line Crossing → Count → Display

Model Used: Custom YOLO model (best.pt) - detects boxes/cartons
Tracking: ByteTrack algorithm for robust object tracking
Counting: Single line crossing detection
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from ultralytics import YOLO
from .box_tracking import ByteTracker, BoxTrack
import json
import os
import torch

# Fix for PyTorch 2.6+ weights_only issue
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

try:
    import ultralytics.nn.tasks as tasks_module
    if hasattr(tasks_module, 'torch_safe_load'):
        _original_torch_safe_load = tasks_module.torch_safe_load
        def _patched_torch_safe_load(weight):
            return torch.load(weight, map_location='cpu', weights_only=False), weight
        tasks_module.torch_safe_load = _patched_torch_safe_load
except:
    pass

# Default model path - change this to use a different model
DEFAULT_MODEL_PATH = r'C:\AlgoOrange Task\RTSP_reader\box_detection_model\box_detection.pt'

# Global variables for interactive line setup
_line_points = []
_setup_window_name = 'Line Setup - Click 2 points'


def load_line_config(video_path: Optional[str] = None) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """Load line configuration from JSON file"""
    config_path = os.path.join(os.path.dirname(__file__), 'line_config.json')
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                if video_path and video_path in config:
                    line_data = config[video_path]
                elif 'default' in config:
                    line_data = config['default']
                else:
                    return None
                return (tuple(line_data['start']), tuple(line_data['end']))
        except Exception as e:
            print(f"[CONFIG] Error loading config: {e}")
    return None


def save_line_config(line_start: Tuple[int, int], line_end: Tuple[int, int], video_path: Optional[str] = None):
    """Save line configuration to JSON file"""
    config_path = os.path.join(os.path.dirname(__file__), 'line_config.json')
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            config = {}
        
        line_data = {'start': list(line_start), 'end': list(line_end)}
        
        if video_path:
            config[video_path] = line_data
        else:
            config['default'] = line_data
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"[CONFIG] Line configuration saved to {config_path}")
    except Exception as e:
        print(f"[CONFIG] Error saving config: {e}")


def line_setup_mouse_callback(event, x, y, flags, param):
    """Mouse callback - captures clicks to define line points"""
    global _line_points
    
    if event == cv2.EVENT_LBUTTONDOWN and len(_line_points) < 2:
        _line_points.append((x, y))
        print(f"[SETUP] Point {len(_line_points)}/2: ({x}, {y})")


def interactive_line_setup(frame: np.ndarray, video_path: Optional[str] = None) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """
    Interactive line setup - click 2 points to define counting line
    
    Args:
        frame: Video frame to display
        video_path: Optional video path for saving config
        
    Returns:
        Tuple of (line_start, line_end) if saved, None if cancelled
    """
    global _line_points
    
    _line_points = []
    display_frame = frame.copy()
    height, width = frame.shape[:2]
    
    # Setup window
    cv2.namedWindow(_setup_window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(_setup_window_name, line_setup_mouse_callback)
    
    print("[SETUP] Line setup started. Click 2 points to define the counting line.")
    print("[SETUP] Controls: 's'=save | 'r'=reset | 'Esc'=cancel")
    
    while True:
        # Redraw frame with current state
        display_frame[:] = frame.copy()
        
        # Draw instruction box
        cv2.rectangle(display_frame, (5, 5), (width-5, 100), (0, 0, 0), -1)
        cv2.rectangle(display_frame, (5, 5), (width-5, 100), (0, 255, 255), 2)
        cv2.putText(display_frame, 'LINE SETUP MODE', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(display_frame, f'Click 2 points ({len(_line_points)}/2)', (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(display_frame, "'s'=save | 'r'=reset | 'Esc'=cancel", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        
        # Draw points and line
        for i, point in enumerate(_line_points):
            cv2.circle(display_frame, point, 8, (0, 255, 255), -1)
            cv2.circle(display_frame, point, 10, (0, 255, 0), 2)
            cv2.putText(display_frame, str(i+1), (point[0]+15, point[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        if len(_line_points) == 2:
            cv2.line(display_frame, _line_points[0], _line_points[1], (0, 255, 255), 3)
        
        cv2.imshow(_setup_window_name, display_frame)
        
        # Handle keyboard input
        key = cv2.waitKey(30) & 0xFF
        
        if key == 27:  # ESC - cancel
            print("[SETUP] Line setup cancelled")
            cv2.destroyWindow(_setup_window_name)
            return None
        
        elif key == ord('r') or key == ord('R'):  # Reset
            _line_points = []
            print("[SETUP] Line reset - click 2 points again")
        
        elif key == ord('s') or key == ord('S'):  # Save
            if len(_line_points) == 2:
                line_start, line_end = _line_points[0], _line_points[1]
                save_line_config(line_start, line_end, video_path)
                cv2.destroyWindow(_setup_window_name)
                print(f"[SETUP] Line saved: {line_start} -> {line_end}")
                return (line_start, line_end)
            else:
                print(f"[SETUP] ERROR: Need exactly 2 points, currently have {len(_line_points)}")


class LineCrossingCounter:
    """
    Line Crossing Counter
    ======================
    Counts boxes when they cross a line.
    
    Logic: Box moves from one side of line to the other = COUNT
    """
    
    def __init__(self, line_start: Tuple[int, int], line_end: Tuple[int, int]):
        self.line_start = line_start
        self.line_end = line_end
        self.boxes_counted = 0
        self.track_sides: Dict[int, str] = {}  # {track_id: 'side1' or 'side2'}
        
        # Calculate line equation: ax + by + c = 0
        x1, y1 = line_start
        x2, y2 = line_end
        self.a = y2 - y1
        self.b = x1 - x2
        self.c = x2 * y1 - x1 * y2
    
    def _get_side(self, point: Tuple[float, float]) -> str:
        """Determine which side of line the point is on"""
        x, y = point
        # Calculate: ax + by + c
        result = self.a * x + self.b * y + self.c
        return 'side1' if result > 0 else 'side2'
    
    def check_crossing(self, track: BoxTrack) -> Optional[str]:
        """
        Check if box crossed the line
        Returns: 'crossed' if box crossed, None otherwise
        """
        if len(track.history) < 2:
            return None
        
        current_position = track.center
        previous_position = track.history[-2]
        
        current_side = self._get_side(current_position)
        previous_side = self._get_side(previous_position)
        
        # First time seeing this box
        if track.track_id not in self.track_sides:
            self.track_sides[track.track_id] = current_side
            return None
        
        # Check for crossing: moved from one side to the other
        if previous_side != current_side:
            # Box crossed the line!
            if track.track_id not in getattr(self, '_counted_tracks', set()):
                self.boxes_counted += 1
                if not hasattr(self, '_counted_tracks'):
                    self._counted_tracks = set()
                self._counted_tracks.add(track.track_id)
                self.track_sides[track.track_id] = current_side
                return 'crossed'
        
        self.track_sides[track.track_id] = current_side
        return None
    
    def draw_line(self, frame: np.ndarray, color: Tuple[int, int, int] = (0, 0, 255), thickness: int = 3):
        """Draw the counting line on frame"""
        cv2.line(frame, self.line_start, self.line_end, color, thickness)
        # Draw arrow to show direction
        mid_x = (self.line_start[0] + self.line_end[0]) // 2
        mid_y = (self.line_start[1] + self.line_end[1]) // 2
        cv2.putText(frame, "COUNTING LINE", (mid_x - 60, mid_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


class BoxCounter:
    """
    Box Counter - Main Class
    =========================
    Detects boxes, tracks them, and counts when they cross a line.
    
    Components:
    1. YOLO model - detects boxes in frames
    2. ByteTracker - tracks boxes across frames
    3. LineCrossingCounter - counts boxes crossing the line
    """
    
    def __init__(self, 
                 model_path: str = DEFAULT_MODEL_PATH,
                 line_start: Optional[Tuple[int, int]] = None,
                 line_end: Optional[Tuple[int, int]] = None,
                 conf_threshold: float = 0.8):
        """
        Initialize Box Counter
        
        Args:
            model_path: Path to YOLO model file
            line_start: Start point of counting line (x, y)
            line_end: End point of counting line (x, y)
            conf_threshold: Minimum confidence to detect box (0.0 to 1.0)
        """
        # Load YOLO model
        print(f"[INIT] Loading YOLO model: {model_path}")
        try:
            self.model = YOLO(model_path)
            print(f"[INIT] Model loaded successfully!")
            
            # Print model classes
            if hasattr(self.model, 'names'):
                print(f"[INIT] Model classes: {self.model.names}")
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            raise
        
        self.conf_threshold = conf_threshold
        
        # Initialize ByteTracker
        print("[INIT] Initializing ByteTracker...")
        self.tracker = ByteTracker(
            max_age=30,
            min_hits=3,
            iou_threshold=0.3,
            score_threshold=conf_threshold
        )
        
        # Initialize line counter
        if line_start and line_end:
            print(f"[INIT] Setting counting line: {line_start} -> {line_end}")
            self.line_counter = LineCrossingCounter(line_start, line_end)
        else:
            self.line_counter = None
        
        self.video_path = None
        print("[INIT] Box counter ready!")
    
    def set_counting_line(self, line_start: Tuple[int, int], line_end: Tuple[int, int]):
        """Set or update the counting line"""
        self.line_counter = LineCrossingCounter(line_start, line_end)
        print(f"[SETUP] Counting line set: {line_start} -> {line_end}")
    
    def _setup_default_line(self, frame_shape: Tuple[int, int], video_path: Optional[str] = None):
        """Setup default counting line if none provided"""
        saved_line = load_line_config(video_path)
        if saved_line:
            line_start, line_end = saved_line
            print(f"[SETUP] Loaded line from config: {line_start} -> {line_end}")
            self.line_counter = LineCrossingCounter(line_start, line_end)
            return
        
        # Default: horizontal line across middle of frame
        height, width = frame_shape[:2]
        margin = int(width * 0.1)
        line_start = (margin, height // 2)
        line_end = (width - margin, height // 2)
        
        self.line_counter = LineCrossingCounter(line_start, line_end)
        print(f"[SETUP] Default line set: {line_start} -> {line_end}")
        print("[SETUP] Press 'z' during playback to configure line interactively")
    
    def detect_boxes(self, frame: np.ndarray) -> List[Tuple[List[float], float]]:
        """Detect boxes in frame using YOLO"""
        # Detect all classes (custom model)
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                score = float(box.conf[0].cpu().numpy())
                detections.append(([float(x1), float(y1), float(x2), float(y2)], score))
        
        return detections
    
    def process_frame(self, frame: np.ndarray, draw: bool = True) -> Dict:
        """
        Process one frame: detect, track, count
        
        Returns:
            Dictionary with counts and annotated frame
        """
        # Setup line if not set
        if self.line_counter is None:
            self._setup_default_line(frame.shape, self.video_path)
        
        # Detect boxes
        detections = self.detect_boxes(frame)
        
        # Track boxes
        active_tracks = self.tracker.update(detections)
        
        # Check line crossings
        crossing_events = []
        for track in active_tracks:
            crossing_result = self.line_counter.check_crossing(track)
            if crossing_result == 'crossed':
                crossing_events.append(track.track_id)
        
        # Draw annotations
        if draw:
            frame = self._draw_annotations(frame, active_tracks, crossing_events)
        
        return {
            'frame': frame,
            'boxes_counted': self.line_counter.boxes_counted,
            'active_tracks': len(active_tracks),
            'crossing_events': crossing_events,
            'detections': len(detections)
        }
    
    def _draw_annotations(self, frame: np.ndarray, tracks: List[BoxTrack], crossing_events: List[int]) -> np.ndarray:
        """Draw boxes, IDs, line, and counts on frame"""
        # Draw counting line
        if self.line_counter:
            self.line_counter.draw_line(frame, color=(0, 0, 255), thickness=3)
        
        # Draw each tracked box
        for track in tracks:
            x1, y1, x2, y2 = [int(coord) for coord in track.bbox]
            
            # Color: yellow if just crossed, blue otherwise
            box_color = (0, 255, 255) if track.track_id in crossing_events else (255, 0, 0)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            
            # Draw center point
            center_x, center_y = [int(c) for c in track.center]
            cv2.circle(frame, (center_x, center_y), 5, box_color, -1)
            
            # Draw track ID
            label = f"ID: {track.track_id}"
            cv2.putText(frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
            
            # Draw "BOX COUNTED!" if just crossed
            if track.track_id in crossing_events:
                cv2.putText(frame, "BOX COUNTED!", (x1, y1 - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Draw count at top
        count_text = f"BOXES COUNTED: {self.line_counter.boxes_counted} | TRACKING: {len(tracks)}"
        cv2.putText(frame, count_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw instructions
        h, w = frame.shape[:2]
        cv2.putText(frame, "Controls: 'z'=line setup | 'r'=reset counts | 'q'=quit | 's'=save line config", 
                   (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        return frame
    
    def reset_counts(self):
        """Reset the box counter"""
        if self.line_counter:
            self.line_counter.boxes_counted = 0
            self.line_counter.track_sides.clear()
            if hasattr(self.line_counter, '_counted_tracks'):
                self.line_counter._counted_tracks.clear()
            print("[RESET] Counts reset")


def process_video(video_path: str, 
                 output_path: Optional[str] = None,
                 line_start: Optional[Tuple[int, int]] = None,
                 line_end: Optional[Tuple[int, int]] = None,
                 conf_threshold: float = 0.8,
                 show_video: bool = True,
                 auto_save: bool = True):
    """
    Process video file and count boxes crossing the line
    
    Args:
        video_path: Path to input video file
        output_path: Path to save output video (optional)
        line_start: Start point of counting line (x, y)
        line_end: End point of counting line (x, y)
        conf_threshold: Minimum confidence to detect box
        show_video: Whether to display video while processing
        auto_save: If True, auto-save to box_loading_detection folder
    """
    video_path_obj = Path(video_path)
    if not video_path_obj.exists():
        print(f"[ERROR] Video file not found: {video_path}")
        return
    
    print(f"[VIDEO] Processing: {video_path_obj.name}")
    
    # Setup output path
    final_output_path = None
    if output_path is None and auto_save:
        box_loading_dir = Path(__file__).parent
        output_filename = f"{video_path_obj.stem}_counted{video_path_obj.suffix}"
        final_output_path = str(box_loading_dir / output_filename)
        print(f"[VIDEO] Auto-saving to: {final_output_path}")
    elif output_path:
        final_output_path = str(Path(output_path))
    
    # Initialize box counter (uses DEFAULT_MODEL_PATH)
    print("[VIDEO] Initializing box counter...")
    print(f"[VIDEO] Using confidence threshold: {conf_threshold}")
    counter = BoxCounter(
        line_start=line_start,
        line_end=line_end,
        conf_threshold=conf_threshold
    )
    counter.video_path = str(video_path_obj)
    
    # Open video
    cap = cv2.VideoCapture(str(video_path_obj))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"[VIDEO] Video: {width}x{height} @ {fps} FPS")
    print("[VIDEO] Controls:")
    print("[VIDEO]   'z' = line setup (click 2 points, then 's' to save)")
    print("[VIDEO]   'r' = reset counts")
    print("[VIDEO]   'q' = quit")
    
    # Setup video writer
    writer = None
    if final_output_path:
        output_path_obj = Path(final_output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(final_output_path, fourcc, fps, (width, height))
    
    # Process frames
    frame_count = 0
    print("[VIDEO] Starting frame processing...\n")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[VIDEO] End of video reached")
                break
            
            frame_count += 1
            
            # Process frame
            result = counter.process_frame(frame, draw=True)
            
            # Save frame
            if writer:
                writer.write(result['frame'])
            
            # Display frame
            if show_video:
                cv2.imshow('Box Loading Detection', result['frame'])
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == ord('Q'):
                    print("[VIDEO] User requested stop")
                    break
                elif key == ord('z') or key == ord('Z'):
                    print("[VIDEO] Entering line setup mode...")
                    new_line = interactive_line_setup(frame.copy(), str(video_path_obj))
                    if new_line:
                        line_start, line_end = new_line
                        counter.set_counting_line(line_start, line_end)
                        save_line_config(line_start, line_end, str(video_path_obj))
                        print("[VIDEO] Line updated and saved!")
                elif key == ord('r') or key == ord('R'):
                    counter.reset_counts()
                    print("[VIDEO] Counts reset")
            
            # Print progress
            if frame_count % 30 == 0:
                print(f"[PROGRESS] Frame {frame_count} | "
                      f"COUNTED: {result['boxes_counted']} | "
                      f"TRACKING: {result['active_tracks']} | "
                      f"DETECTIONS: {result['detections']}")
    
    except KeyboardInterrupt:
        print("\n[VIDEO] Stopped by user (Ctrl+C)")
    
    finally:
        cap.release()
        if writer:
            writer.release()
            print(f"[VIDEO] Processed video saved: {final_output_path}")
        cv2.destroyAllWindows()
        
        # Print final results
        print("\n" + "="*60)
        print("FINAL RESULTS:")
        print("="*60)
        print(f"  Boxes Counted:       {counter.line_counter.boxes_counted}")
        if final_output_path:
            print(f"  Output Video:        {final_output_path}")
        print("="*60)


def process_stream(stream_source: str,
                   line_start: Optional[Tuple[int, int]] = None,
                   line_end: Optional[Tuple[int, int]] = None,
                   conf_threshold: float = 0.8,
                   show_video: bool = True):
    """
    Process live stream (RTSP or camera) and count boxes crossing the line
    
    Args:
        stream_source: RTSP URL or camera index (0, 1, 2, ...)
        line_start: Start point of counting line (x, y)
        line_end: End point of counting line (x, y)
        conf_threshold: Minimum confidence to detect box
        show_video: Whether to display video
    """
    print("[STREAM] Initializing box counter...")
    print(f"[STREAM] Using confidence threshold: {conf_threshold}")
    counter = BoxCounter(
        line_start=line_start,
        line_end=line_end,
        conf_threshold=conf_threshold
    )
    counter.video_path = stream_source
    
    # Convert camera index string to int if needed
    if isinstance(stream_source, str) and stream_source.isdigit():
        stream_source = int(stream_source)
    
    # Open stream
    cap = cv2.VideoCapture(stream_source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open stream: {stream_source}")
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    
    print(f"[STREAM] Stream: {width}x{height} @ {fps} FPS")
    print("[STREAM] Controls:")
    print("[STREAM]   'z' = line setup (click 2 points, then 's' to save)")
    print("[STREAM]   'r' = reset counts")
    print("[STREAM]   'q' = quit\n")
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[STREAM] Warning: Failed to read frame. Retrying...")
                continue
            
            frame_count += 1
            
            # Process frame
            result = counter.process_frame(frame, draw=True)
            
            # Display frame
            if show_video:
                cv2.imshow('Box Loading Detection - Live Stream', result['frame'])
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == ord('Q'):
                    print("[STREAM] User requested stop")
                    break
                elif key == ord('z') or key == ord('Z'):
                    print("[STREAM] Entering line setup mode...")
                    new_line = interactive_line_setup(frame.copy(), str(stream_source))
                    if new_line:
                        line_start, line_end = new_line
                        counter.set_counting_line(line_start, line_end)
                        save_line_config(line_start, line_end, str(stream_source))
                        print("[STREAM] Line updated and saved!")
                elif key == ord('r') or key == ord('R'):
                    counter.reset_counts()
                    print("[STREAM] Counts reset")
            
            # Print counts
            if frame_count % 30 == 0:
                print(f"[STREAM] Frame {frame_count} | "
                      f"COUNTED: {result['boxes_counted']} | "
                      f"TRACKING: {result['active_tracks']}")
    
    except KeyboardInterrupt:
        print("\n[STREAM] Stopped by user (Ctrl+C)")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        print("\n" + "="*60)
        print("FINAL RESULTS:")
        print("="*60)
        print(f"  Boxes Counted:       {counter.line_counter.boxes_counted}")
        print("="*60)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        base_dir = Path(__file__).parent.parent
        sample_dir = base_dir / "sample_video"
        
        if sample_dir.exists():
            video_files = list(sample_dir.glob("*.mp4")) + list(sample_dir.glob("*.avi"))
            if video_files:
                video_path = str(video_files[0])
                print(f"Using video: {video_path}")
            else:
                print(f"No video files found in {sample_dir}")
                sys.exit(1)
        else:
            print(f"sample_video folder not found at {sample_dir}")
            print("Usage: python box_count.py <video_path>")
            sys.exit(1)
    
    process_video(video_path, show_video=True)
