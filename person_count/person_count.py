"""
STEP 2: PERSON COUNTING SYSTEM
===============================
This is the MAIN system that combines everything:
1. YOLO detects people in each frame
2. Tracker follows each person across frames
3. Line counter detects when people cross the line

FLOW: Video Frame → Detect → Track → Check Crossing → Count → Display
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from ultralytics import YOLO
from .object_tracking import SimpleByteTracker, Track
import json
import os

# Global variables for interactive line setup
_setup_mode = False
_line_points = []
_setup_frame = None
_setup_window_name = 'Line Setup - Click 2 points'


def load_line_config(video_path: Optional[str] = None) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """Load line configuration from JSON file"""
    config_path = os.path.join(os.path.dirname(__file__), 'line_config.json')
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                # If video-specific config exists, use it; otherwise use default
                if video_path and video_path in config:
                    line_data = config[video_path]
                elif 'default' in config:
                    line_data = config['default']
                else:
                    return None
                # Convert list of lists to tuples
                line_start = tuple(line_data['start'])
                line_end = tuple(line_data['end'])
                return (line_start, line_end)
        except Exception as e:
            print(f"[CONFIG] Error loading config: {e}")
    return None


def save_line_config(line_start: Tuple[int, int], line_end: Tuple[int, int], video_path: Optional[str] = None):
    """Save line configuration to JSON file"""
    config_path = os.path.join(os.path.dirname(__file__), 'line_config.json')
    try:
        # Load existing config if it exists
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            config = {}
        
        # Convert tuples to lists for JSON serialization
        line_data = {
            'start': [int(line_start[0]), int(line_start[1])],
            'end': [int(line_end[0]), int(line_end[1])]
        }
        
        # Save video-specific or default config
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
    """Mouse callback for interactive line setup"""
    global _line_points, _setup_frame
    
    if event == cv2.EVENT_LBUTTONDOWN:
        _line_points.append((x, y))
        print(f"Point {len(_line_points)}: ({x}, {y})")
        
        # Draw point on frame
        cv2.circle(_setup_frame, (x, y), 5, (0, 255, 255), -1)
        
        # Draw line if we have 2 points
        if len(_line_points) == 2:
            cv2.line(_setup_frame, _line_points[0], _line_points[1], (0, 255, 255), 2)
            print("Line complete! Press 's' to save, 'r' to reset, or 'Esc' to cancel.")
        
        cv2.imshow(_setup_window_name, _setup_frame)


def interactive_line_setup(frame: np.ndarray, video_path: Optional[str] = None) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """
    Interactive line setup - click 2 points to define counting line
    
    Args:
        frame: Video frame to display
        video_path: Optional video path for video-specific config
        
    Returns:
        Tuple of ((x1, y1), (x2, y2)) or None if cancelled
    """
    global _setup_mode, _line_points, _setup_frame
    
    _setup_mode = True
    _line_points = []
    _setup_frame = frame.copy()
    
    h, w = frame.shape[:2]
    
    # Display instructions
    cv2.putText(_setup_frame, 'LINE SETUP MODE', (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(_setup_frame, 'Click 2 points to define counting line', (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(_setup_frame, "Press 's' to save | 'r' to reset | 'Esc' to cancel", (10, 90), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    
    cv2.namedWindow(_setup_window_name)
    cv2.setMouseCallback(_setup_window_name, line_setup_mouse_callback)
    
    cv2.imshow(_setup_window_name, _setup_frame)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC - cancel
            print("[SETUP] Line setup cancelled")
            cv2.destroyWindow(_setup_window_name)
            _setup_mode = False
            return None
        
        elif key == ord('r') or key == ord('R'):  # Reset
            _line_points = []
            _setup_frame = frame.copy()
            cv2.putText(_setup_frame, 'LINE SETUP MODE', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(_setup_frame, 'Click 2 points to define counting line', (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(_setup_frame, "Press 's' to save | 'r' to reset | 'Esc' to cancel", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(_setup_window_name, _setup_frame)
            print("[SETUP] Line reset - click 2 points again")
        
        elif key == ord('s') or key == ord('S'):  # Save
            if len(_line_points) == 2:
                line_start = _line_points[0]
                line_end = _line_points[1]
                save_line_config(line_start, line_end, video_path)
                cv2.destroyWindow(_setup_window_name)
                _setup_mode = False
                print(f"[SETUP] Line saved: {line_start} to {line_end}")
                return (line_start, line_end)
            else:
                print(f"[SETUP] Need 2 points, currently have {len(_line_points)}")
    
    _setup_mode = False
    return None


class LineCrossingCounter:
    """
    STEP 2.1: LINE CROSSING COUNTER
    ================================
    This counts when people cross a virtual line.
    
    RULES:
    - Person moves from ABOVE line to BELOW line = ENTRY (coming IN)
    - Person moves from BELOW line to ABOVE line = EXIT (going OUT)
    
    For overhead camera view:
    - Moving DOWN (towards camera) = IN
    - Moving UP (away from camera) = OUT
    """
    
    def __init__(self, line_start: Tuple[int, int], line_end: Tuple[int, int]):
        """
        STEP 2.1.1: INITIALIZE LINE COUNTER
        Set up the counting line and reset counters.
        """
        # Store line endpoints
        self.line_start = line_start  # (x, y) start point
        self.line_end = line_end      # (x, y) end point
        
        # Initialize counters
        self.entry_count = 0  # How many people entered
        self.exit_count = 0   # How many people exited
        
        # Calculate the Y coordinate of the line (for horizontal lines)
        # This is the threshold: above this = "above", below this = "below"
        self.line_y = (line_start[1] + line_end[1]) / 2
        
        # Remember which side each person is on
        # Format: {track_id: 'above' or 'below'}
        self.track_sides: Dict[int, str] = {}
        
    def _get_side(self, point: Tuple[float, float]) -> str:
        """
        STEP 2.1.2: DETERMINE WHICH SIDE OF LINE
        Check if a point is above or below the line.
        
        Returns: 'above' or 'below'
        """
        point_y = point[1]  # Get Y coordinate
        
        # If Y is greater than line_y, point is below the line
        # If Y is less than line_y, point is above the line
        if point_y > self.line_y:
            return 'below'
        else:
            return 'above'
    
    def check_crossing(self, track: Track) -> Optional[str]:
        """
        STEP 2.1.3: CHECK IF PERSON CROSSED THE LINE
        ==============================================
        This is called for EVERY person in EVERY frame.
        
        PROCESS:
        1. Get current position and previous position
        2. Check which side of line each is on
        3. If they moved from one side to the other = CROSSING!
        4. Update count and return 'entry' or 'exit'
        
        Returns:
            'entry' if person crossed IN (above → below)
            'exit' if person crossed OUT (below → above)
            None if no crossing happened
        """
        # Need at least 2 positions to detect movement
        if len(track.history) < 2:
            return None
        
        # STEP 1: Get current and previous positions
        current_position = track.center  # Where person is NOW
        previous_position = track.history[-2]  # Where person was BEFORE
        
        # STEP 2: Determine which side of line each position is on
        current_side = self._get_side(current_position)
        previous_side = self._get_side(previous_position)
        
        # STEP 3: First time seeing this person? Just remember their side
        if track.track_id not in self.track_sides:
            self.track_sides[track.track_id] = current_side
            return None  # No crossing yet
        
        # STEP 4: CHECK FOR CROSSING
        # Case 1: Person moved from ABOVE to BELOW = ENTRY (coming IN)
        if previous_side == 'above' and current_side == 'below':
            # Make sure we're tracking them as being above before
            if self.track_sides.get(track.track_id) == 'above':
                self.track_sides[track.track_id] = 'below'  # Update side
                self.entry_count += 1  # Increment entry counter
                return 'entry'  # Return crossing event
        
        # Case 2: Person moved from BELOW to ABOVE = EXIT (going OUT)
        elif previous_side == 'below' and current_side == 'above':
            # Make sure we're tracking them as being below before
            if self.track_sides.get(track.track_id) == 'below':
                self.track_sides[track.track_id] = 'above'  # Update side
                self.exit_count += 1  # Increment exit counter
                return 'exit'  # Return crossing event
        
        # STEP 5: No crossing, just update which side they're on
        self.track_sides[track.track_id] = current_side
        return None
    
    def draw_line(self, frame: np.ndarray, color: Tuple[int, int, int] = (0, 0, 255), thickness: int = 2):
        """
        STEP 2.1.4: DRAW THE COUNTING LINE
        Draw a red line on the frame so we can see where counting happens.
        """
        cv2.line(frame, self.line_start, self.line_end, color, thickness)


class PersonCounter:
    """
    STEP 2.2: MAIN PERSON COUNTER CLASS
    ====================================
    This is the MAIN class that brings everything together.
    
    COMPONENTS:
    1. YOLO model - detects people in frames
    2. Tracker - follows each person across frames
    3. Line counter - counts when people cross the line
    
    MAIN FUNCTION: process_frame() - processes one frame at a time
    """
    
    def __init__(self, 
                 model_path: str = 'yolov8l.pt',
                 line_start: Optional[Tuple[int, int]] = None,
                 line_end: Optional[Tuple[int, int]] = None,
                 conf_threshold: float = 0.5):
        """
        STEP 2.2.1: INITIALIZE THE COUNTER
        Set up YOLO model, tracker, and prepare for counting.
        """
        # STEP 1: Load YOLO model (this detects people)
        print(f"[INIT] Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)  # This will download model if needed
        self.conf_threshold = conf_threshold  # Minimum confidence to count as person
        
        # STEP 2: Initialize tracker (this follows people across frames)
        print("[INIT] Initializing tracker...")
        self.tracker = SimpleByteTracker(
            max_age=30,        # Delete track if not seen for 30 frames
            min_hits=3,        # Need 3 detections to confirm a person
            iou_threshold=0.3,  # How much boxes must overlap to match
            score_threshold=conf_threshold
        )
        
        # STEP 3: Store line settings (will be set up on first frame)
        self.line_start = line_start
        self.line_end = line_end
        self.line_counter: Optional[LineCrossingCounter] = None
        self.frame_shape = None
        self.video_path = None  # Will be set when processing video
        
        print("[INIT] Person counter ready!")
        print("[INIT] Note: If line not set, you can set it manually, use default, or press 'z' during video playback for interactive setup")
        
    def _setup_line(self, frame_shape: Tuple[int, int], video_path: Optional[str] = None):
        """
        STEP 2.2.2: SETUP COUNTING LINE
        Create the counting line on the first frame.
        First tries to load from config file, then uses default if not provided.
        """
        # Try to load from config file first
        saved_line = load_line_config(video_path)
        if saved_line:
            self.line_start, self.line_end = saved_line
            print(f"[SETUP] Loaded line from config: {self.line_start} to {self.line_end}")
            self.line_counter = LineCrossingCounter(self.line_start, self.line_end)
            self.frame_shape = frame_shape
            return
        
        height, width = frame_shape[:2]
        
        # If user didn't specify line position, use default
        if self.line_start is None or self.line_end is None:
            print("[SETUP] Using default line position (horizontal, 1/3 from bottom)")
            margin = 50  # Leave 50 pixels margin on sides
            line_y = int(height * 0.67)  # 67% down from top = 1/3 from bottom
            self.line_start = (margin, line_y)
            self.line_end = (width - margin, line_y)
        else:
            print(f"[SETUP] Using custom line: {self.line_start} to {self.line_end}")
        
        # Create the line counter
        self.line_counter = LineCrossingCounter(self.line_start, self.line_end)
        self.frame_shape = frame_shape
        print("[SETUP] Press 'z' during playback to configure line interactively")
    
    def set_line(self, line_start: Tuple[int, int], line_end: Tuple[int, int]):
        """
        Set or update the counting line coordinates.
        
        Args:
            line_start: (x, y) start point of the line
            line_end: (x, y) end point of the line
        """
        self.line_start = line_start
        self.line_end = line_end
        if self.frame_shape:
            self.line_counter = LineCrossingCounter(line_start, line_end)
            print(f"[SETUP] Line updated: {line_start} to {line_end}")
        
    def detect_persons(self, frame: np.ndarray) -> List[Tuple[List[float], float]]:
        """
        STEP 2.2.3: DETECT PEOPLE IN FRAME
        ===================================
        Use YOLO to find all people in the current frame.
        
        PROCESS:
        1. Run YOLO on frame
        2. Filter for class 0 (person class)
        3. Extract bounding boxes and confidence scores
        4. Return list of detections
        
        Returns: List of (bounding_box, confidence_score)
        """
        # Run YOLO detection
        # classes=[0] means only detect class 0 (person)
        # conf=threshold means only detections above confidence threshold
        results = self.model(frame, classes=[0], conf=self.conf_threshold, verbose=False)
        
        # Extract detections from results
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get bounding box: [left_x, top_y, right_x, bottom_y]
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Get confidence score (how sure YOLO is this is a person)
                score = float(box.conf[0].cpu().numpy())
                
                # Store detection
                detections.append(([float(x1), float(y1), float(x2), float(y2)], score))
        
        return detections
    
    def process_frame(self, frame: np.ndarray, draw: bool = True) -> Dict:
        """
        STEP 2.2.4: MAIN PROCESSING FUNCTION
        ====================================
        This is the CORE function called for EVERY frame.
        
        PROCESS FLOW:
        1. Setup line (first frame only)
        2. Detect people in frame (YOLO)
        3. Track people across frames (Tracker)
        4. Check if anyone crossed the line (Line Counter)
        5. Draw annotations on frame (optional)
        6. Return results
        
        Args:
            frame: Current video frame (image)
            draw: Whether to draw boxes, IDs, counts on frame
        
        Returns: Dictionary with counts and annotated frame
        """
        # STEP 1: SETUP LINE (only on first frame)
        if self.line_counter is None:
            print("[PROCESS] Setting up counting line...")
            self._setup_line(frame.shape, self.video_path)
        
        # STEP 2: DETECT PEOPLE
        # Use YOLO to find all people in this frame
        detections = self.detect_persons(frame)
        
        # STEP 3: TRACK PEOPLE
        # Match detections to existing tracks, create new tracks for new people
        active_tracks = self.tracker.update(detections)
        
        # STEP 4: CHECK LINE CROSSINGS
        # For each tracked person, check if they crossed the line
        crossing_events = []
        for track in active_tracks:
            crossing_result = self.line_counter.check_crossing(track)
            if crossing_result:  # If they crossed (returned 'entry' or 'exit')
                crossing_events.append((track.track_id, crossing_result))
        
        # STEP 5: DRAW ANNOTATIONS
        # Draw boxes, IDs, line, counts on the frame
        if draw:
            frame = self._draw_annotations(frame, active_tracks, crossing_events)
        
        # STEP 6: RETURN RESULTS
        return {
            'frame': frame,  # Frame with annotations drawn
            'entry_count': self.line_counter.entry_count,  # Total people entered
            'exit_count': self.line_counter.exit_count,    # Total people exited
            'total_in': self.line_counter.entry_count,     # Same as entry_count
            'total_out': self.line_counter.exit_count,      # Same as exit_count
            'current_count': self.line_counter.entry_count - self.line_counter.exit_count,  # Net count
            'active_tracks': len(active_tracks)  # How many people currently tracked
        }
    
    def _draw_annotations(self, 
                         frame: np.ndarray, 
                         tracks: List[Track],
                         crossing_events: List[Tuple[int, str]]) -> np.ndarray:
        """
        STEP 2.2.5: DRAW ANNOTATIONS ON FRAME
        =====================================
        Draw visual information on the frame:
        - Red counting line
        - Bounding boxes around people
        - Track IDs
        - Center points
        - Crossing events (ENTERED/EXITED)
        - Counts (IN/OUT/CURRENT)
        """
        # STEP 1: Draw the counting line in red
        if self.line_counter:
            self.line_counter.draw_line(frame, color=(0, 0, 255), thickness=2)
        
        # STEP 2: Draw each tracked person
        for track in tracks:
            # Get bounding box coordinates
            x1, y1, x2, y2 = [int(coord) for coord in track.bbox]
            
            # Check if this person just crossed the line
            just_crossed = any(track_id == track.track_id for track_id, _ in crossing_events)
            
            # Choose color: Yellow if just crossed, Blue otherwise
            if just_crossed:
                box_color = (0, 255, 255)  # Yellow (BGR format)
            else:
                box_color = (255, 0, 0)  # Blue (BGR format)
            
            # Draw bounding box around person
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            
            # Draw center point (small circle)
            center_x, center_y = [int(c) for c in track.center]
            cv2.circle(frame, (center_x, center_y), 5, box_color, -1)
            
            # Draw track ID label (e.g., "ID: 5")
            label = f"ID: {track.track_id}"
            cv2.putText(frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
            
            # Draw crossing event text (ENTERED or EXITED)
            for track_id, event_type in crossing_events:
                if track_id == track.track_id:
                    if event_type == 'entry':
                        event_text = "ENTERED"
                    else:
                        event_text = "EXITED"
                    cv2.putText(frame, event_text, (x1, y1 - 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # STEP 3: Draw counts at top of frame
        if self.line_counter:
            count_text = (
                f"IN: {self.line_counter.entry_count} | "
                f"OUT: {self.line_counter.exit_count} | "
                f"CURRENT: {self.line_counter.entry_count - self.line_counter.exit_count}"
            )
            cv2.putText(frame, count_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw instructions
        h, w = frame.shape[:2]
        cv2.putText(frame, "Press 'z' for line setup | 'r' to reset counts | 'q' to quit", 
                   (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        return frame
    
    def reset_counts(self):
        """Reset entry and exit counts."""
        if self.line_counter:
            self.line_counter.entry_count = 0
            self.line_counter.exit_count = 0
            self.line_counter.track_sides.clear()


def process_video(video_path: str, 
                 output_path: Optional[str] = None,
                 model_path: str = 'yolov8l.pt',
                 line_start: Optional[Tuple[int, int]] = None,
                 line_end: Optional[Tuple[int, int]] = None,
                 show_video: bool = True,
                 auto_save: bool = True):
    """
    STEP 3: PROCESS VIDEO FILE
    ===========================
    Main function to process a video file and count people.
    
    PROCESS:
    1. Open video file
    2. Initialize counter
    3. Loop through each frame
    4. Process each frame (detect → track → count)
    5. Display/save results
    6. Print final counts
    
    Args:
        auto_save: If True and output_path not provided, auto-save to person_count folder
    """
    # STEP 1: Validate video file exists
    video_path_obj = Path(video_path)
    if not video_path_obj.exists():
        print(f"[ERROR] Video file not found: {video_path}")
        return
    
    print(f"[VIDEO] Processing: {video_path_obj.name}")
    
    # STEP 2: Setup output path (auto-save to person_count folder if not specified)
    final_output_path = None
    if output_path is None and auto_save:
        # Get person_count folder path
        person_count_dir = Path(__file__).parent
        # Create output filename: original_name_counted.mp4
        output_filename = f"{video_path_obj.stem}_counted{video_path_obj.suffix}"
        final_output_path = str(person_count_dir / output_filename)
        print(f"[VIDEO] Auto-saving processed video to: {final_output_path}")
    elif output_path:
        # User specified custom output path
        final_output_path = str(Path(output_path))
        print(f"[VIDEO] Will save output to: {final_output_path}")
    else:
        # Don't save video
        final_output_path = None
        print(f"[VIDEO] Video will not be saved (set auto_save=True or provide output_path)")
    
    # STEP 3: Initialize the person counter
    print("[VIDEO] Initializing counter...")
    counter = PersonCounter(
        model_path=model_path,
        line_start=line_start,
        line_end=line_end
    )
    counter.video_path = str(video_path_obj)  # Store video path for config
    
    # STEP 4: Open video file
    cap = cv2.VideoCapture(str(video_path_obj))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return
    
    # STEP 5: Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"[VIDEO] Video properties: {width}x{height} @ {fps} FPS")
    print("[VIDEO] Controls: 'z' = line setup, 'r' = reset counts, 'q' = quit")
    
    # STEP 6: Setup video writer (if saving output)
    writer = None
    if final_output_path:
        # Ensure output directory exists
        output_path_obj = Path(final_output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"[VIDEO] Saving processed video to: {final_output_path}")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(final_output_path, fourcc, fps, (width, height))
    
    # STEP 6: MAIN LOOP - Process each frame
    frame_count = 0
    print("[VIDEO] Starting frame processing...")
    print("[VIDEO] Press 'q' to quit\n")
    
    try:
        while True:
            # Read next frame from video
            ret, frame = cap.read()
            if not ret:
                print("[VIDEO] End of video reached")
                break
            
            frame_count += 1
            
            # Process this frame (detect → track → count)
            result = counter.process_frame(frame, draw=True)
            
            # Save frame to output video (if specified)
            if writer:
                writer.write(result['frame'])
            
            # Display frame in window
            if show_video:
                cv2.imshow('Person Counting', result['frame'])
                # Check keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("[VIDEO] User requested stop")
                    break
                elif key == ord('z') or key == ord('Z'):
                    # Enter line setup mode
                    print("[VIDEO] Entering line setup mode...")
                    new_line = interactive_line_setup(frame, str(video_path_obj))
                    if new_line:
                        counter.set_line(new_line[0], new_line[1])
                        print("[VIDEO] Line updated! Processing continues...")
                elif key == ord('r') or key == ord('R'):
                    # Reset counts
                    counter.reset_counts()
                    print("[VIDEO] Counts reset")
            
            # Print progress every 30 frames
            if frame_count % 30 == 0:
                print(f"[PROGRESS] Frame {frame_count} | "
                      f"IN: {result['entry_count']} | "
                      f"OUT: {result['exit_count']} | "
                      f"Current: {result['current_count']}")
    
    except KeyboardInterrupt:
        print("\n[VIDEO] Stopped by user (Ctrl+C)")
    
    finally:
        # STEP 7: CLEANUP
        print("[VIDEO] Cleaning up...")
        cap.release()  # Close video file
        if writer:
            writer.release()  # Close output video
            print(f"[VIDEO] Processed video saved successfully!")
        cv2.destroyAllWindows()  # Close display windows
        
        # STEP 8: Print final counts
        final_result = counter.process_frame(
            np.zeros((height, width, 3), dtype=np.uint8), 
            draw=False
        )
        print("\n" + "="*60)
        print("FINAL RESULTS:")
        print("="*60)
        print(f"  People Entered (IN):  {final_result['entry_count']}")
        print(f"  People Exited (OUT):  {final_result['exit_count']}")
        print(f"  Current Inside:       {final_result['current_count']}")
        if final_output_path:
            print(f"  Output Video:          {final_output_path}")
        print("="*60)


def process_stream(stream_source: str,
                   model_path: str = 'yolov8l.pt',
                   line_start: Optional[Tuple[int, int]] = None,
                   line_end: Optional[Tuple[int, int]] = None,
                   show_video: bool = True):
    """
    STEP 4: PROCESS LIVE STREAM
    ============================
    Process RTSP stream or camera feed in real-time.
    
    Same as process_video() but for live streams (keeps running until stopped).
    """
    # STEP 1: Initialize counter
    print("[STREAM] Initializing counter...")
    counter = PersonCounter(
        model_path=model_path,
        line_start=line_start,
        line_end=line_end
    )
    counter.video_path = str(stream_source)  # Store stream source for config
    
    # STEP 2: Convert camera index string to int if needed
    if isinstance(stream_source, str) and stream_source.isdigit():
        stream_source = int(stream_source)
    
    # STEP 3: Open stream (RTSP URL or camera)
    cap = cv2.VideoCapture(stream_source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open stream: {stream_source}")
        return
    
    # STEP 4: Get stream properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    
    print(f"[STREAM] Stream properties: {width}x{height} @ {fps} FPS")
    print("[STREAM] Press 'q' to quit\n")
    
    # STEP 5: MAIN LOOP - Process frames continuously
    frame_count = 0
    
    try:
        while True:
            # Read next frame
            ret, frame = cap.read()
            if not ret:
                print("[STREAM] Warning: Failed to read frame. Retrying...")
                continue
            
            frame_count += 1
            
            # Process frame
            result = counter.process_frame(frame, draw=True)
            
            # Display frame
            if show_video:
                cv2.imshow('Person Counting - Live Stream', result['frame'])
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("[STREAM] User requested stop")
                    break
                elif key == ord('z') or key == ord('Z'):
                    # Enter line setup mode
                    print("[STREAM] Entering line setup mode...")
                    new_line = interactive_line_setup(frame, str(stream_source))
                    if new_line:
                        counter.set_line(new_line[0], new_line[1])
                        print("[STREAM] Line updated! Processing continues...")
                elif key == ord('r') or key == ord('R'):
                    # Reset counts
                    counter.reset_counts()
                    print("[STREAM] Counts reset")
            
            # Print counts every 30 frames
            if frame_count % 30 == 0:
                print(f"[STREAM] Frame {frame_count} | "
                      f"IN: {result['entry_count']} | "
                      f"OUT: {result['exit_count']} | "
                      f"Current: {result['current_count']}")
    
    except KeyboardInterrupt:
        print("\n[STREAM] Stopped by user (Ctrl+C)")
    
    finally:
        # STEP 6: CLEANUP
        cap.release()
        cv2.destroyAllWindows()
        
        # Print final counts
        print("\n" + "="*60)
        print("FINAL RESULTS:")
        print("="*60)
        print(f"  People Entered (IN):  {result['entry_count']}")
        print(f"  People Exited (OUT):  {result['exit_count']}")
        print(f"  Current Inside:       {result['current_count']}")
        print("="*60)



#########################################################################




if __name__ == "__main__":
    import sys
    
    # Example usage
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        # Default: look for video in sample_video folder
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
            print("Usage: python person_count.py <video_path>")
            sys.exit(1)
    
    # Process video
    process_video(video_path, show_video=True)
