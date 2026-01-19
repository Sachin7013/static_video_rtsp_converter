"""
TEST SCRIPT FOR BOX LOADING DETECTION
======================================
This script tests the box loading detection system on the sample video.

USAGE:
    python test_box_loading.py

This will:
1. Load the video from sample_video folder
2. Detect boxes using YOLOv11
3. Track boxes using ByteTrack
4. Count boxes as they enter the truck loading zone
5. Display results and save annotated video
"""

from pathlib import Path
from box_loading_detection.box_count import BoxCounter, process_video
import cv2


def test_with_default_zone():
    """
    Test 1: Process video with default loading zone
    The system will automatically create a zone on the left side of the frame.
    """
    print("="*60)
    print("TEST 1: Processing with default loading zone")
    print("="*60)
    
    # Find video in sample_video folder
    base_dir = Path(__file__).parent
    sample_dir = base_dir / "sample_video"
    
    if not sample_dir.exists():
        print(f"Error: sample_video folder not found at {sample_dir}")
        return
    
    video_files = list(sample_dir.glob("*.mp4")) + list(sample_dir.glob("*.avi"))
    if not video_files:
        print(f"Error: No video files found in {sample_dir}")
        return
    
    video_path = str(video_files[0])
    print(f"Processing: {video_path}")
    
    # Process video with default zone
    process_video(
        video_path=video_path,
        show_video=True  # Display video while processing
    )


def test_with_custom_zone():
    """
    Test 2: Process video with custom loading zone coordinates
    You can manually define the zone where boxes enter the truck.
    
    Format: [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    This defines a rectangle/polygon for the truck loading area.
    """
    print("="*60)
    print("TEST 2: Processing with custom loading zone")
    print("="*60)
    
    base_dir = Path(__file__).parent
    sample_dir = base_dir / "sample_video"
    
    video_files = list(sample_dir.glob("*.mp4")) + list(sample_dir.glob("*.avi"))
    if not video_files:
        print(f"Error: No video files found in {sample_dir}")
        return
    
    video_path = str(video_files[0])
    
    # CUSTOM ZONE COORDINATES
    # Adjust these based on your video!
    # Format: [top_left, top_right, bottom_right, bottom_left]
    # For the container loading video, the zone should be around the container opening
    # You can get these coordinates by:
    # 1. Running test_with_interactive_zone() first
    # 2. Or manually inspecting the video frame
    
    # Example: Zone on left side (typical truck position)
    # These are percentages - adjust based on your video resolution
    # For 1280x720 video, this would be approximately:
    custom_zone = [
        (50, 100),    # Top-left corner
        (500, 100),   # Top-right corner
        (500, 500),   # Bottom-right corner
        (50, 500)     # Bottom-left corner
    ]
    
    print(f"Using custom zone: {custom_zone}")
    print("Note: You may need to adjust these coordinates for your specific video")
    
    process_video(
        video_path=video_path,
        zone_coords=custom_zone,
        show_video=True
    )


def test_with_interactive_zone():
    """
    Test 3: Interactive zone selection
    This allows you to click on the video to define the loading zone.
    """
    print("="*60)
    print("TEST 3: Interactive zone selection")
    print("="*60)
    print("Instructions:")
    print("1. A video frame will be displayed")
    print("2. Click 4 points to define the loading zone rectangle")
    print("3. Press 'Enter' to confirm, 'Esc' to cancel")
    print("="*60)
    
    base_dir = Path(__file__).parent
    sample_dir = base_dir / "sample_video"
    
    video_files = list(sample_dir.glob("*.mp4")) + list(sample_dir.glob("*.avi"))
    if not video_files:
        print(f"Error: No video files found in {sample_dir}")
        return
    
    video_path = str(video_files[0])
    
    # Open video to get first frame
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        print(f"Error: Cannot read video: {video_path}")
        cap.release()
        return
    
    cap.release()
    
    # Interactive zone selection
    points = []
    
    def mouse_callback(event, x, y, flags, param):
        """Mouse callback to capture clicks"""
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            print(f"Point {len(points)}: ({x}, {y})")
            
            # Draw point on frame
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            if len(points) > 1:
                # Draw line connecting points
                cv2.line(frame, points[-2], points[-1], (0, 255, 0), 2)
            cv2.imshow('Select Loading Zone (Click 4 points)', frame)
    
    cv2.namedWindow('Select Loading Zone (Click 4 points)')
    cv2.setMouseCallback('Select Loading Zone (Click 4 points)', mouse_callback)
    
    # Display frame and wait for 4 clicks
    cv2.imshow('Select Loading Zone (Click 4 points)', frame)
    print("\nClick 4 points to define the loading zone rectangle...")
    
    while len(points) < 4:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC to cancel
            print("Cancelled")
            cv2.destroyAllWindows()
            return
        if key == 13:  # Enter to finish early
            if len(points) >= 3:
                break
    
    cv2.destroyAllWindows()
    
    if len(points) < 3:
        print("Error: Need at least 3 points")
        return
    
    # If 3 points, create rectangle
    if len(points) == 3:
        # Create rectangle from 3 points
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        zone = [
            (min(x_coords), min(y_coords)),  # Top-left
            (max(x_coords), min(y_coords)),  # Top-right
            (max(x_coords), max(y_coords)),  # Bottom-right
            (min(x_coords), max(y_coords))   # Bottom-left
        ]
    else:
        zone = points[:4]
    
    print(f"\nZone selected: {zone}")
    print("Processing video with selected zone...\n")
    
    # Process video with selected zone
    process_video(
        video_path=video_path,
        zone_coords=zone,
        show_video=True
    )


def test_frame_by_frame():
    """
    Test 4: Frame-by-frame processing (for custom integration)
    This shows how to process frames one at a time for custom applications.
    """
    print("="*60)
    print("TEST 4: Frame-by-frame processing")
    print("="*60)
    
    base_dir = Path(__file__).parent
    sample_dir = base_dir / "sample_video"
    
    video_files = list(sample_dir.glob("*.mp4")) + list(sample_dir.glob("*.avi"))
    if not video_files:
        print(f"Error: No video files found in {sample_dir}")
        return
    
    video_path = str(video_files[0])
    
    # Initialize counter (uses DEFAULT_MODEL_PATH)
    counter = BoxCounter()
    # model = YOLO(r'C:\AlgoOrange Task\RTSP_reader\CementBag\Box-Detection-in-Warehouse-using-Vision-Based-ML-Engineering\best.pt')
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video: {video_path}")
        return
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process frame
            result = counter.process_frame(frame, draw=True)
            
            # Access results
            boxes_loaded = result['boxes_loaded']
            active_tracks = result['active_tracks']
            zone_entries = result['zone_entries']
            
            # Display
            cv2.imshow('Box Loading Detection', result['frame'])
            
            # Print every 30 frames
            if frame_count % 30 == 0:
                print(f"Frame {frame_count}: "
                      f"LOADED={boxes_loaded}, "
                      f"TRACKING={active_tracks}, "
                      f"NEW_ENTRIES={len(zone_entries)}")
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\nStopped by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        # Final counts
        print(f"\nFinal - Boxes Loaded: {counter.zone_detector.boxes_loaded}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("BOX LOADING DETECTION - TEST SCRIPT")
    print("="*60)
    print("\nChoose a test:")
    print("1. Default zone (automatic)")
    print("2. Custom zone (manual coordinates)")
    print("3. Interactive zone selection (click to define)")
    print("4. Frame-by-frame processing")
    print("\nRunning Test 1 (Default Zone) by default...\n")
    
    # Run test 1 by default
    test_with_default_zone()
    
    # Uncomment to run other tests:
    # test_with_custom_zone()
    # test_with_interactive_zone()
    # test_frame_by_frame()
