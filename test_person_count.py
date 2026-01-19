"""
================================================================================
PERSON COUNTING SYSTEM - TEST SCRIPT
================================================================================

HOW TO USE:
1. Put your test video in the 'sample_video' folder
2. Run this script: python test_person_count.py
3. Watch the video with counting overlay
4. Press 'q' to quit

WHAT IT DOES:
- Finds video in sample_video folder
- Processes each frame: Detect → Track → Count
- Shows real-time counts on screen
- Prints final counts when done

CUSTOMIZATION:
You can customize the counting line position by modifying the process_video() call.
"""

import sys
from pathlib import Path

# Add parent directory to path (so we can import person_count module)
sys.path.insert(0, str(Path(__file__).parent))

from person_count.person_count import process_video


def main():
    """
    MAIN FUNCTION
    =============
    This is the entry point when you run: python test_person_count.py
    """
    print("\n" + "="*70)
    print("PERSON COUNTING SYSTEM - TEST")
    print("="*70)
    
    # STEP 1: Find the sample_video folder
    base_dir = Path(__file__).parent
    sample_dir = base_dir / "sample_video"
    
    if not sample_dir.exists():
        print(f"[ERROR] sample_video folder not found at: {sample_dir}")
        print("[INFO] Please create the folder and add your test video files.")
        return
    
    # STEP 2: Find video files in the folder
    video_files = (
        list(sample_dir.glob("*.mp4")) + 
        list(sample_dir.glob("*.avi")) + 
        list(sample_dir.glob("*.mov")) + 
        list(sample_dir.glob("*.mkv"))
    )
    
    if not video_files:
        print(f"[ERROR] No video files found in: {sample_dir}")
        print("[INFO] Supported formats: .mp4, .avi, .mov, .mkv")
        return
    
    # STEP 3: Use the first video found
    video_path = str(video_files[0])
    print(f"[INFO] Found video: {video_path}")
    print(f"[INFO] Press 'q' in the video window to quit")
    print("="*70 + "\n")
    
    # STEP 4: Process the video
    # This will:
    #   1. Load YOLO model
    #   2. Open video file
    #   3. Process each frame (detect → track → count)
    #   4. Display results
    #   5. Save processed video to person_count folder (auto-save enabled)
    #   6. Print final counts
    
    # DEFAULT: Uses horizontal line 1/3 from bottom
    # To customize line position, uncomment and modify:
    # line_start = (50, 200)   # (x, y) start point
    # line_end = (600, 200)    # (x, y) end point
    
    process_video(
        video_path=video_path,
        show_video=True,  # Set to False to process without displaying
        model_path='yolov8l.pt',  # Will download automatically if not exists
        auto_save=True,  # Automatically save processed video to person_count folder
        # line_start=line_start,  # Uncomment to use custom line
        # line_end=line_end,       # Uncomment to use custom line
        # output_path="custom_path.mp4",  # Uncomment to specify custom output path
    )


if __name__ == "__main__":
    # This runs when you execute: python test_person_count.py
    main()
