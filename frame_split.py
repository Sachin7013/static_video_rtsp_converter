import cv2
import os
from pathlib import Path

def video_to_frames(video_path, output_folder, frames_per_second=3, use_first_word=False):
    # Create output folder if not exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Cannot open video file")
        return

    # Extract video name for frame naming
    video_file = Path(video_path)
    video_name = video_file.stem  # Get filename without extension
    
    # Option to use first word only
    if use_first_word:
        # Get first word (split by underscore, space, or hyphen)
        first_word = video_name.replace("_", " ").replace("-", " ").split()[0] if video_name else "video"
        video_prefix = first_word
    else:
        # Use full video name (sanitized for filename)
        video_prefix = video_name.replace(" ", "_").replace("-", "_")
        # Remove special characters, keep only alphanumeric and underscores
        video_prefix = "".join(c if c.isalnum() or c == "_" else "" for c in video_prefix)
    
    print(f"Video name: {video_name}")
    print(f"Using prefix for frames: {video_prefix}")

    # Get video FPS
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video FPS: {fps}")
    
    # Calculate frame interval to get desired frames per second
    # If we want 3 frames per second, we need to skip frames
    frame_interval = int(fps / frames_per_second)
    if frame_interval < 1:
        frame_interval = 1
    
    print(f"Extracting {frames_per_second} frames per second (every {frame_interval} frame(s))")

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Video end
        
        frame_count += 1
        
        # Only save frames at the calculated interval (3 frames per second)
        if frame_count % frame_interval == 0:
            saved_count += 1
            # Frame name based on video name or first word
            frame_name = f"{video_prefix}_frame_{saved_count:04d}.jpg"
            frame_path = os.path.join(output_folder, frame_name)
            cv2.imwrite(frame_path, frame)

    cap.release()
    print(f"Done! Total frames processed: {frame_count}, Frames saved: {saved_count}")

# ---- Usage ----
video_path = r"C:\AlgoOrange Task\RTSP_reader\sample_video\Wild_shootout_in_Memphis_caught_on_video_with_child_in_the_middle_of_it_720P.mp4"
output_folder = "output_frames"       # Frames save folder

# Use full video name in frame names (default)
video_to_frames(video_path, output_folder)

# Or use only first word of video name
# video_to_frames(video_path, output_folder, use_first_word=True)