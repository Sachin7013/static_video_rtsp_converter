import sys
from pathlib import Path
import cv2
from ultralytics import YOLO
from datetime import datetime
import numpy as np
import torch

# Configuration
# MODEL_NAME = "yolov8x-pose-p6.pt"
MODEL_NAME = "yolov8l-pose.pt"  # YOLOv8 pose model name (yolov8n-pose.pt, yolov8s-pose.pt, yolov8m-pose.pt, yolov8l-pose.pt, yolov8x-pose.pt)
PERSON_CONFIDENCE_THRESHOLD = 0.7 # Person detection confidence >= 0.7
POSE_DETECTION_CONFIDENCE_THRESHOLD = 0.7 # Pose detection confidence >= 0.7
KEYPOINT_VISIBILITY_THRESHOLD = 0.5
SAHI_SIZE_MODE = False  # Set True to enable SAHI slicing, False to disable
SAHI_SLICE_COUNT = 4  # Number of slices to split frame into (4 = 2x2, 9 = 3x3, 16 = 4x4, etc.)
WRIST_SHOULDER_TOLERANCE_PERCENT = 0.20  # Allow wrist to be 20% below shoulder and still trigger (0.20 = 20%)
VIDEO_RECORD_DURATION = 5  # Duration in seconds to record when gun detection is suspected

# YOLO Pose keypoint indices
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10

VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".avi", ".flv", ".ts", ".m4v", ".wmv", ".webm", ".mpg", ".mpeg")


def log(msg: str):
    print(msg, flush=True)


def is_keypoint_visible(keypoint):
    """Check if keypoint is visible."""
    if keypoint is None or len(keypoint) < 3:
        return False
    return keypoint[2] > KEYPOINT_VISIBILITY_THRESHOLD


def calculate_tolerance(shoulder, elbow=None):
    """Calculate tolerance based on body proportions."""
    if elbow is not None and is_keypoint_visible(elbow):
        # Use elbow-to-shoulder distance as reference (typically ~20-30% of arm length)
        shoulder_y = shoulder[1]
        elbow_y = elbow[1]
        arm_segment = abs(shoulder_y - elbow_y)
        # Allow 20% of arm segment below shoulder
        return arm_segment * WRIST_SHOULDER_TOLERANCE_PERCENT
    else:
        # Fallback: use fixed tolerance (20 pixels)
        return 20  # pixels


def check_point_above_shoulder(shoulder, point, tolerance):
    """Check if a point (wrist or elbow) is at or above shoulder level, with tolerance.
    
    Args:
        shoulder: Shoulder keypoint [x, y, confidence]
        point: Point keypoint (wrist or elbow) [x, y, confidence]
        tolerance: Tolerance value in pixels
    
    Returns:
        True if point is at shoulder level, above shoulder, or within tolerance below shoulder
    """
    if not is_keypoint_visible(shoulder) or not is_keypoint_visible(point):
        return False
    
    shoulder_y = shoulder[1]
    point_y = point[1]
    
    # Point triggers alert if:
    # - At shoulder level (point_y == shoulder_y)
    # - Above shoulder (point_y < shoulder_y)
    # - Within tolerance below shoulder (point_y <= shoulder_y + tolerance)
    return point_y <= (shoulder_y + tolerance)


def check_wrist_above_shoulder(shoulder, wrist, elbow=None):
    """Check if wrist is at or above shoulder level, with tolerance for slightly below shoulder.
    
    Args:
        shoulder: Shoulder keypoint [x, y, confidence]
        wrist: Wrist keypoint [x, y, confidence]
        elbow: Optional elbow keypoint [x, y, confidence] for calculating tolerance
    
    Returns:
        True if wrist is at shoulder level, above shoulder, or within tolerance below shoulder
    """
    tolerance = calculate_tolerance(shoulder, elbow)
    return check_point_above_shoulder(shoulder, wrist, tolerance)


def check_elbow_above_shoulder(shoulder, elbow):
    """Check if elbow is at or above shoulder level, with tolerance for slightly below shoulder.
    
    Args:
        shoulder: Shoulder keypoint [x, y, confidence]
        elbow: Elbow keypoint [x, y, confidence]
    
    Returns:
        True if elbow is at shoulder level, above shoulder, or within tolerance below shoulder
    """
    # For elbow, we can use a reference point or fixed tolerance
    # Since we don't have another reference, use fixed tolerance
    tolerance = calculate_tolerance(shoulder, elbow)
    return check_point_above_shoulder(shoulder, elbow, tolerance)


def detect_gun_suspicion(keypoints):
    """Detect if person has wrist or elbow at or near shoulder level (gun suspicion).
    
    Triggers alert when wrist OR elbow is:
    - At shoulder level (straight)
    - Above shoulder
    - Within tolerance below shoulder (~80% of way to shoulder)
    Works for both left and right hands.
    """
    if keypoints is None or len(keypoints) < 17:
        return False
    
    # Check left arm
    left_shoulder = keypoints[LEFT_SHOULDER]
    left_elbow = keypoints[LEFT_ELBOW] if len(keypoints) > LEFT_ELBOW else None
    left_wrist = keypoints[LEFT_WRIST] if len(keypoints) > LEFT_WRIST else None
    
    # Check left wrist
    if left_wrist is not None and check_wrist_above_shoulder(left_shoulder, left_wrist, left_elbow):
        return True
    
    # Check left elbow
    if left_elbow is not None and check_elbow_above_shoulder(left_shoulder, left_elbow):
        return True
    
    # Check right arm
    right_shoulder = keypoints[RIGHT_SHOULDER]
    right_elbow = keypoints[RIGHT_ELBOW] if len(keypoints) > RIGHT_ELBOW else None
    right_wrist = keypoints[RIGHT_WRIST] if len(keypoints) > RIGHT_WRIST else None
    
    # Check right wrist
    if right_wrist is not None and check_wrist_above_shoulder(right_shoulder, right_wrist, right_elbow):
        return True
    
    # Check right elbow
    if right_elbow is not None and check_elbow_above_shoulder(right_shoulder, right_elbow):
        return True
    
    return False


def split_frame_into_slices(frame, slice_count):
    """Split frame into configurable number of slices for SAHI.
    
    Args:
        frame: Input frame to split
        slice_count: Number of slices to create (should be a perfect square: 4, 9, 16, 25, etc.)
    
    Returns:
        List of tuples: (slice_image, x_offset, y_offset)
    """
    height, width = frame.shape[:2]
    
    # Calculate grid dimensions (rows x cols)
    # For perfect squares: 4=2x2, 9=3x3, 16=4x4, etc.
    grid_size = int(np.sqrt(slice_count))
    if grid_size * grid_size != slice_count:
        # If not perfect square, round to nearest and adjust
        grid_size = int(np.ceil(np.sqrt(slice_count)))
        log(f"Warning: SAHI_SLICE_COUNT={slice_count} is not a perfect square. Using {grid_size}x{grid_size} grid ({grid_size*grid_size} slices)")
    
    rows = grid_size
    cols = grid_size
    
    # Calculate slice dimensions
    slice_width = width // cols
    slice_height = height // rows
    
    slices = []
    
    # Create slices in row-major order
    for row in range(rows):
        for col in range(cols):
            # Calculate slice boundaries
            x_start = col * slice_width
            x_end = width if col == cols - 1 else (col + 1) * slice_width
            y_start = row * slice_height
            y_end = height if row == rows - 1 else (row + 1) * slice_height
            
            # Extract slice
            slice_img = frame[y_start:y_end, x_start:x_end]
            
            # Store slice with offset coordinates
            slices.append((slice_img, x_start, y_start))
    
    return slices


def remove_duplicate_detections(all_detections, distance_threshold=100):
    """Remove duplicate detections from overlapping slices."""
    if len(all_detections) <= 1:
        return all_detections
    
    filtered = []
    used = [False] * len(all_detections)
    
    for i in range(len(all_detections)):
        if used[i]:
            continue
        
        kp_i, conf_i = all_detections[i]
        visible_i = kp_i[kp_i[:, 2] > KEYPOINT_VISIBILITY_THRESHOLD]
        if len(visible_i) == 0:
            continue
        
        center_i = (visible_i[:, 0].mean(), visible_i[:, 1].mean())
        
        for j in range(i + 1, len(all_detections)):
            if used[j]:
                continue
            
            kp_j, conf_j = all_detections[j]
            visible_j = kp_j[kp_j[:, 2] > KEYPOINT_VISIBILITY_THRESHOLD]
            if len(visible_j) == 0:
                continue
            
            center_j = (visible_j[:, 0].mean(), visible_j[:, 1].mean())
            dist = np.sqrt((center_i[0] - center_j[0])**2 + (center_i[1] - center_j[1])**2)
            
            if dist < distance_threshold:
                # Keep the one with higher confidence
                if conf_i >= conf_j:
                    used[j] = True
                else:
                    used[i] = True
                    break
        
        if not used[i]:
            filtered.append(all_detections[i])
    
    return filtered


def detect_with_sahi(pose_model, frame):
    """Detect using SAHI (configurable slices) or normal detection."""
    if not SAHI_SIZE_MODE:
        # Normal detection
        return pose_model(frame, conf=PERSON_CONFIDENCE_THRESHOLD, verbose=False)
    
    height, width = frame.shape[:2]
    
    # For very small frames, use normal detection
    if width < 640 or height < 480:
        return pose_model(frame, conf=PERSON_CONFIDENCE_THRESHOLD, verbose=False)
    
    # Split frame into configurable number of slices
    slices = split_frame_into_slices(frame, SAHI_SLICE_COUNT)
    all_detections = []
    
    # Process each slice
    for slice_img, x_offset, y_offset in slices:
        slice_results = pose_model(slice_img, conf=PERSON_CONFIDENCE_THRESHOLD, verbose=False)
        
        for result in slice_results:
            if result.keypoints is None or len(result.keypoints.data) == 0:
                continue
            
            # Get person confidences
            person_confidences = None
            if hasattr(result, 'boxes') and result.boxes is not None:
                try:
                    person_confidences = result.boxes.conf.cpu().numpy()
                except:
                    pass
            
            keypoints_data = result.keypoints.data.cpu().numpy()
            
            for idx, person_kp in enumerate(keypoints_data):
                # Filter by person confidence
                if person_confidences is not None and idx < len(person_confidences):
                    if person_confidences[idx] < PERSON_CONFIDENCE_THRESHOLD:
                        continue
                
                # Transform keypoints to original frame coordinates
                transformed_kp = person_kp.copy()
                transformed_kp[:, 0] += x_offset
                transformed_kp[:, 1] += y_offset
                
                confidence = person_confidences[idx] if person_confidences is not None else 0.7
                all_detections.append((transformed_kp, confidence))
    
    # Remove duplicates
    filtered_detections = remove_duplicate_detections(all_detections)
    
    if len(filtered_detections) == 0:
        return []
    
    # Create result object with merged detections
    all_keypoints = np.stack([det[0] for det in filtered_detections])
    all_confidences = np.array([det[1] for det in filtered_detections])
    
    # Run normal detection to get result structure
    base_result = pose_model(frame, conf=PERSON_CONFIDENCE_THRESHOLD, verbose=False)
    
    if base_result and len(base_result) > 0:
        result = base_result[0]
        result.keypoints.data = torch.from_numpy(all_keypoints).float()
        result.orig_img = frame
        
        # Update boxes if available
        if hasattr(result, 'boxes') and result.boxes is not None:
            try:
                # Create boxes from keypoints
                boxes = []
                for kp in all_keypoints:
                    visible_kp = kp[kp[:, 2] > KEYPOINT_VISIBILITY_THRESHOLD]
                    if len(visible_kp) > 0:
                        x1 = max(0, visible_kp[:, 0].min() - 10)
                        y1 = max(0, visible_kp[:, 1].min() - 10)
                        x2 = visible_kp[:, 0].max() + 10
                        y2 = visible_kp[:, 1].max() + 10
                        boxes.append([x1, y1, x2, y2])
                    else:
                        boxes.append([0, 0, 0, 0])
                
                result.boxes.xyxy = torch.from_numpy(np.array(boxes)).float()
                result.boxes.conf = torch.from_numpy(all_confidences).float()
            except:
                pass
        
        return [result]
    
    return []


def load_pose_model(base_dir):
    """Load YOLOv11 Pose model. Model will be auto-downloaded if not found locally."""
    # First check if model exists locally
    model_path = base_dir / MODEL_NAME
    if model_path.exists():
        log(f"Loading pose model from local file: {model_path}")
        model = YOLO(str(model_path))
    else:
        # Let Ultralytics auto-download the model if not found locally
        log(f"Model not found locally. Loading from Ultralytics (will auto-download if needed): {MODEL_NAME}")
        model = YOLO(MODEL_NAME)
    log("Model loaded successfully!")
    return model


def find_videos(source_dir):
    """Find all video files in directory."""
    videos = []
    for ext in VIDEO_EXTS:
        videos.extend(sorted(source_dir.glob(f"*{ext}")))
    return sorted(videos)


def process_video(video_path, pose_model):
    """Process video and detect gun suspicion."""
    log(f"\nProcessing: {video_path.name}")
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log(f"ERROR: Cannot open {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    log(f"Resolution: {width}x{height}, FPS: {fps:.1f}")
    if SAHI_SIZE_MODE:
        grid_size = int(np.sqrt(SAHI_SLICE_COUNT))
        log(f"SAHI Mode: ENABLED ({SAHI_SLICE_COUNT} slices = {grid_size}x{grid_size} grid)")
    else:
        log(f"SAHI Mode: DISABLED")
    
    # Create output folder for detected frames and videos
    output_dir = Path("GunDetected") / video_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    
    frame_number = 0
    saved_count = 0
    
    # Video recording variables
    video_writer = None
    recording_frames = 0
    recording_frames_total = int(fps * VIDEO_RECORD_DURATION)
    is_recording = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            # Stop any ongoing recording before looping
            if is_recording and video_writer is not None:
                video_writer.release()
                video_writer = None
                is_recording = False
                log(f"  Stopped recording due to video loop")
            
            # Loop video
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_number = 0
            continue
        
        frame_number += 1
        
        # Run pose detection (with SAHI if enabled)
        results = detect_with_sahi(pose_model, frame)
        
        # Draw pose keypoints
        annotated_frame = frame.copy()
        if results and len(results) > 0:
            annotated_frame = results[0].plot()
        
        gun_detected = False
        max_confidence = 0.0
        
        # Process each detection
        for result in results:
            if result.keypoints is None or len(result.keypoints.data) == 0:
                continue
            
            # Get person confidences
            person_confidences = None
            if hasattr(result, 'boxes') and result.boxes is not None:
                try:
                    person_confidences = result.boxes.conf.cpu().numpy()
                except:
                    pass
            
            # Get keypoints
            keypoints_data = result.keypoints.data.cpu().numpy()
            
            # Process each person
            for person_idx, person_keypoints in enumerate(keypoints_data):
                # Check person confidence >= 0.7
                if person_confidences is not None and person_idx < len(person_confidences):
                    if person_confidences[person_idx] < PERSON_CONFIDENCE_THRESHOLD:
                        continue
                
                # Check if gun suspicion detected
                if detect_gun_suspicion(person_keypoints):
                    # Get person confidence
                    person_confidence = 0.0
                    if person_confidences is not None and person_idx < len(person_confidences):
                        person_confidence = person_confidences[person_idx]
                    
                    # Only trigger alert if confidence is above threshold
                    if person_confidence >= POSE_DETECTION_CONFIDENCE_THRESHOLD:
                        gun_detected = True
                        # Store max confidence for saving
                        max_confidence = max(max_confidence, person_confidence)
                        
                        # Get bounding box
                        if hasattr(result, 'boxes') and result.boxes is not None:
                            try:
                                boxes = result.boxes.xyxy.cpu().numpy()
                                if person_idx < len(boxes):
                                    box = boxes[person_idx]
                                    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                                    
                                    # Draw alert box
                                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                                    
                                    # Draw alert text with confidence score
                                    label = f"GUN DETECTION SUSPECTED (Conf: {person_confidence:.2f})"
                                    font_scale = 0.7
                                    thickness = 2
                                    (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                                    
                                    # Background for text
                                    cv2.rectangle(annotated_frame, 
                                                (x1, y1 - text_h - 10), 
                                                (x1 + text_w + 10, y1), 
                                                (0, 0, 255), -1)
                                    
                                    # Text
                                    cv2.putText(annotated_frame, label, (x1 + 5, y1 - 5), 
                                              cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
                            except:
                                pass
        
        # Display status
        status = f"Frame: {frame_number}"
        if gun_detected:
            status += " [GUN DETECTION SUSPECTED]"
        cv2.putText(annotated_frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Saved: {saved_count} frames", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Save processed frame if gun detected (after all annotations are done)
        if gun_detected:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{video_path.stem}_gun_detected_conf_{max_confidence:.2f}_{timestamp}.jpg"
            output_path = output_dir / output_filename
            cv2.imwrite(str(output_path), annotated_frame)
            saved_count += 1
            log(f"  Frame {frame_number}: GUN DETECTION SUSPECTED - Saved to {output_path.name}")
            
            # Start video recording if not already recording
            if not is_recording:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                video_filename = f"{video_path.stem}_gun_detected_conf_{max_confidence:.2f}_{timestamp}.mp4"
                video_path_out = output_dir / video_filename
                
                # Initialize video writer
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(str(video_path_out), fourcc, fps, (width, height))
                is_recording = True
                recording_frames = 0
                log(f"  Started recording: {video_filename}")
        
        # Record frame if recording
        if is_recording and video_writer is not None:
            video_writer.write(annotated_frame)
            recording_frames += 1
            
            # Stop recording after duration
            if recording_frames >= recording_frames_total:
                video_writer.release()
                video_writer = None
                is_recording = False
                log(f"  Saved {VIDEO_RECORD_DURATION} second video evidence")
                recording_frames = 0
        
        # Show live stream
        cv2.imshow(f"Gun Detection: {video_path.stem}", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        if frame_number % 100 == 0:
            log(f"  Processed {frame_number} frames, Saved {saved_count} frames")
    
    # Stop any ongoing recording
    if is_recording and video_writer is not None:
        video_writer.release()
        video_writer = None
        is_recording = False
        log(f"  Stopped recording at end of video")
    
    cap.release()
    cv2.destroyAllWindows()
    log(f"Completed! Saved {saved_count} frames with gun detection")


def main():
    """Main function."""
    base_dir = Path(__file__).resolve().parent
    source_dir = base_dir / "sample_video"
    
    if not source_dir.exists():
        log(f"ERROR: Folder 'sample_video' not found at {source_dir}")
        log("Creating sample_video folder...")
        source_dir.mkdir(exist_ok=True)
        log(f"Please add video files to: {source_dir}")
        sys.exit(1)
    
    # Load model
    try:
        pose_model = load_pose_model(base_dir)
    except Exception as e:
        log(f"ERROR: Failed to load model: {e}")
        sys.exit(1)
    
    # Find videos
    video_paths = find_videos(source_dir)
    if not video_paths:
        log(f"ERROR: No video files found in {source_dir}")
        sys.exit(1)
    
    log(f"Found {len(video_paths)} video(s)")
    
    # Process videos
    for video_path in video_paths:
        try:
            process_video(video_path, pose_model)
        except KeyboardInterrupt:
            log("\nStopped by user")
            break
        except Exception as e:
            log(f"ERROR processing {video_path.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    log("\nProcessing completed!")


if __name__ == "__main__":
    main()
