"""
BOX TRACKING MODULE - ByteTrack Implementation
===============================================
This module tracks each box/package across video frames using ByteTrack algorithm.
Each box gets a unique ID that stays the same as it moves.

FLOW: Detection → Track → Update Track → Match to Next Frame

ByteTrack Algorithm:
1. First Association: Match high-confidence detections to existing tracks
2. Second Association: Match low-confidence detections to unmatched tracks
3. Track Management: Create new tracks, update existing, remove stale tracks
"""

import numpy as np
from typing import List, Tuple, Optional


class BoxTrack:
    """
    STEP 1: BOX TRACK CLASS
    =======================
    This represents ONE box being tracked.
    Stores: position, ID, movement history, state (outside/entering/inside/counted)
    """
    
    def __init__(self, track_id: int, bbox: List[float], score: float, frame_id: int):
        """
        STEP 1.1: CREATE A NEW TRACK
        When we first detect a box, create a BoxTrack object for it.
        
        Args:
            track_id: Unique ID for this box (0, 1, 2, 3, ...)
            bbox: Bounding box [left_x, top_y, right_x, bottom_y]
            score: Confidence score from YOLO (0.0 to 1.0)
            frame_id: Frame number where we first saw this box
        """
        # Store the box's unique ID (0, 1, 2, 3, ...)
        self.track_id = track_id
        
        # Bounding box: [left_x, top_y, right_x, bottom_y]
        self.bbox = bbox
        
        # How confident YOLO is that this is a box (0.0 to 1.0)
        self.score = score
        
        # Which frame number we first saw this box
        self.frame_id = frame_id
        
        # Calculate center point of the bounding box
        self.center = self._calculate_center()
        
        # Store all past positions (to see movement direction)
        self.history = [self.center]
        
        # How many frames this track has existed
        self.age = 1
        
        # How many times we've detected this box in a row
        self.hit_streak = 1
        
        # Track state: 'outside', 'entering', 'inside', 'counted'
        # This helps us know if the box has been counted
        self.state = 'outside'
        
        # Whether this box has been counted (to prevent double-counting)
        self.is_counted = False
        
    def _calculate_center(self) -> Tuple[float, float]:
        """
        STEP 1.2: FIND CENTER POINT
        Calculate the center (x, y) of the bounding box.
        This is what we use to track movement and check zone entry.
        
        Returns:
            (center_x, center_y): Center coordinates of the box
        """
        x1, y1, x2, y2 = self.bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        return (center_x, center_y)
    
    def update(self, bbox: List[float], score: float, frame_id: int):
        """
        STEP 1.3: UPDATE TRACK WITH NEW POSITION
        When we see this box again in the next frame, update its position.
        
        Args:
            bbox: New bounding box [left_x, top_y, right_x, bottom_y]
            score: New confidence score
            frame_id: Current frame number
        """
        # Update bounding box to new position
        self.bbox = bbox
        self.score = score
        self.frame_id = frame_id
        
        # Calculate new center point
        self.center = self._calculate_center()
        
        # Add new position to history (so we can see movement)
        self.history.append(self.center)
        
        # Increase counters
        self.age += 1
        self.hit_streak += 1
        
        # Keep only last 50 positions (to save memory, but more than person tracking)
        # Boxes move faster, so we need more history
        if len(self.history) > 50:
            self.history = self.history[-50:]
    
    def predict(self) -> Tuple[float, float]:
        """
        STEP 1.4: PREDICT NEXT POSITION
        Estimate where this box will be in the next frame based on movement.
        This helps with tracking when boxes move quickly.
        
        Returns:
            (predicted_x, predicted_y): Predicted center position
        """
        if len(self.history) < 2:
            return self.center
        
        # Calculate movement speed (velocity)
        last_x, last_y = self.history[-1]
        prev_x, prev_y = self.history[-2]
        dx = last_x - prev_x  # movement in x direction
        dy = last_y - prev_y  # movement in y direction
        
        # Predict next position (assume constant velocity)
        predicted_x = self.center[0] + dx
        predicted_y = self.center[1] + dy
        return (predicted_x, predicted_y)


class ByteTracker:
    """
    STEP 2: BYTETRACK TRACKER CLASS
    ================================
    This manages ALL tracks (all boxes being tracked).
    
    MAIN JOB: Match new detections to existing tracks using ByteTrack algorithm.
    - If detection matches existing track → Update that track
    - If detection is new → Create new track
    - If track not seen for too long → Delete track
    
    ByteTrack Strategy:
    1. First association: Match high-confidence detections to confirmed tracks
    2. Second association: Match remaining detections to unconfirmed tracks
    3. Create new tracks for unmatched detections
    """
    
    def __init__(self, 
                 max_age: int = 30,
                 min_hits: int = 3,
                 iou_threshold: float = 0.3,
                 score_threshold: float = 0.4):
        """
        STEP 2.1: INITIALIZE TRACKER
        Set up the tracker with configuration values optimized for boxes.
        
        Args:
            max_age: If we don't see a box for this many frames, delete its track
            min_hits: Box must be detected this many times to be "confirmed"
            iou_threshold: How much boxes must overlap to be considered the same (0.0 to 1.0)
            score_threshold: Minimum confidence score to consider a detection (0.0 to 1.0)
        """
        # If we don't see a box for 30 frames, delete their track
        self.max_age = max_age
        
        # Box must be detected 3 times in a row to be "confirmed"
        # This prevents false detections from being counted
        self.min_hits = min_hits
        
        # How much boxes must overlap to be considered the same box (0.0 to 1.0)
        # Lower = more lenient matching (good for fast-moving boxes)
        self.iou_threshold = iou_threshold
        
        # Minimum confidence score to consider a detection (0.0 to 1.0)
        # Lower for boxes because they vary more in appearance than people
        self.score_threshold = score_threshold
        
        # List of all tracks (all boxes we're tracking)
        self.tracks: List[BoxTrack] = []
        
        # Counter to give each box a unique ID (0, 1, 2, 3, ...)
        self.track_id_counter = 0
        
        # Current frame number
        self.frame_id = 0
        
    def _iou(self, box1: List[float], box2: List[float]) -> float:
        """
        STEP 2.2: CALCULATE IoU (Intersection over Union)
        ===================================================
        This measures how much two boxes overlap.
        Returns: 0.0 (no overlap) to 1.0 (perfect overlap)
        
        IoU Formula:
        IoU = (Area of Intersection) / (Area of Union)
        
        Example:
        - IoU = 0.9 → Very likely the same box
        - IoU = 0.1 → Probably different boxes
        
        Args:
            box1: First bounding box [x1, y1, x2, y2]
            box2: Second bounding box [x1, y1, x2, y2]
        
        Returns:
            iou_score: Overlap score between 0.0 and 1.0
        """
        # Extract coordinates from both boxes
        x1_box1, y1_box1, x2_box1, y2_box1 = box1
        x1_box2, y1_box2, x2_box2, y2_box2 = box2
        
        # STEP 1: Find the overlapping area (intersection)
        # The intersection is the area where both boxes overlap
        overlap_left = max(x1_box1, x1_box2)    # Left edge of overlap
        overlap_top = max(y1_box1, y1_box2)     # Top edge of overlap
        overlap_right = min(x2_box1, x2_box2)   # Right edge of overlap
        overlap_bottom = min(y2_box1, y2_box2)  # Bottom edge of overlap
        
        # Check if boxes actually overlap
        # If right < left or bottom < top, there's no overlap
        if overlap_right < overlap_left or overlap_bottom < overlap_top:
            return 0.0  # No overlap
        
        # Calculate intersection area (the overlapping rectangle)
        intersection_area = (overlap_right - overlap_left) * (overlap_bottom - overlap_top)
        
        # Calculate area of each box
        area_box1 = (x2_box1 - x1_box1) * (y2_box1 - y1_box1)
        area_box2 = (x2_box2 - x1_box2) * (y2_box2 - y1_box2)
        
        # Calculate union (total area covered by both boxes)
        # Union = Area1 + Area2 - Intersection (to avoid double-counting)
        union_area = area_box1 + area_box2 - intersection_area
        
        # Avoid division by zero
        if union_area == 0:
            return 0.0
        
        # IoU = intersection / union
        iou_score = intersection_area / union_area
        return iou_score
    
    def _match_detections_to_tracks(self, 
                                    detections: List[Tuple[List[float], float]],
                                    tracks: List[BoxTrack]) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        STEP 2.3: MATCH DETECTIONS TO TRACKS
        ====================================
        This is the CORE matching logic of ByteTrack.
        
        GOAL: Figure out which detection belongs to which existing track.
        
        PROCESS:
        1. Calculate IoU between every detection and every track
        2. Find potential matches (where IoU > threshold)
        3. Sort by IoU score (best matches first)
        4. Greedy matching: match best overlaps first
        5. Return: matched pairs, unmatched detections, unmatched tracks
        
        Args:
            detections: List of (bounding_box, confidence_score)
            tracks: List of existing BoxTrack objects
        
        Returns:
            matched_pairs: List of (detection_idx, track_idx) tuples
            unmatched_detection_indices: List of detection indices that didn't match
            unmatched_track_indices: List of track indices that didn't match
        """
        # EDGE CASE 1: No detections → all tracks are unmatched
        if len(detections) == 0:
            unmatched_track_indices = list(range(len(tracks)))
            return [], [], unmatched_track_indices
        
        # EDGE CASE 2: No tracks → all detections are unmatched
        if len(tracks) == 0:
            unmatched_detection_indices = list(range(len(detections)))
            return [], unmatched_detection_indices, []
        
        # STEP 1: Calculate IoU for every detection-track pair
        # Create a matrix: rows = detections, columns = tracks
        iou_matrix = np.zeros((len(detections), len(tracks)))
        
        for detection_idx, (detection_bbox, _) in enumerate(detections):
            for track_idx, track in enumerate(tracks):
                # Calculate how much this detection overlaps with this track
                overlap_score = self._iou(detection_bbox, track.bbox)
                iou_matrix[detection_idx, track_idx] = overlap_score
        
        # STEP 2: Find all potential matches (where IoU > threshold)
        potential_matches = []
        for detection_idx in range(len(detections)):
            for track_idx in range(len(tracks)):
                overlap_score = iou_matrix[detection_idx, track_idx]
                if overlap_score > self.iou_threshold:
                    # Store: (detection_index, track_index, overlap_score)
                    potential_matches.append((detection_idx, track_idx, overlap_score))
        
        # STEP 3: Sort by overlap score (highest first)
        # This ensures we match the best overlaps first (greedy approach)
        potential_matches.sort(key=lambda x: x[2], reverse=True)
        
        # STEP 4: Greedy matching - match best overlaps first
        matched_pairs = []  # List of (detection_idx, track_idx)
        used_detection_indices = set()
        used_track_indices = set()
        unmatched_track_indices = list(range(len(tracks)))
        
        for detection_idx, track_idx, _ in potential_matches:
            # Only match if neither detection nor track is already used
            # This ensures one detection matches one track (no duplicates)
            if detection_idx not in used_detection_indices and track_idx not in used_track_indices:
                matched_pairs.append((detection_idx, track_idx))
                used_detection_indices.add(detection_idx)
                used_track_indices.add(track_idx)
                # Remove from unmatched tracks
                if track_idx in unmatched_track_indices:
                    unmatched_track_indices.remove(track_idx)
        
        # STEP 5: Find unmatched detections
        unmatched_detection_indices = [
            idx for idx in range(len(detections)) 
            if idx not in used_detection_indices
        ]
        
        return matched_pairs, unmatched_detection_indices, unmatched_track_indices
    
    def update(self, detections: List[Tuple[List[float], float]]) -> List[BoxTrack]:
        """
        STEP 2.4: MAIN UPDATE FUNCTION (BYTETRACK ALGORITHM)
        =====================================================
        This is called for EVERY frame. This implements the ByteTrack algorithm.
        
        BYTETRACK PROCESS:
        1. Filter detections (remove low confidence)
        2. Separate tracks into confirmed and unconfirmed
        3. FIRST ASSOCIATION: Match high-confidence detections to confirmed tracks
        4. SECOND ASSOCIATION: Match remaining detections to unconfirmed tracks
        5. Create new tracks for unmatched detections
        6. Remove old tracks that haven't been seen
        
        Args:
            detections: List of (bounding_box, confidence_score) from YOLO
        
        Returns:
            List of confirmed active tracks (boxes we're currently tracking)
        """
        # Move to next frame
        self.frame_id += 1
        
        # STEP 1: FILTER DETECTIONS
        # Only keep detections with confidence above threshold
        high_confidence_detections = [
            (bbox, score) for bbox, score in detections 
            if score >= self.score_threshold
        ]
        
        # Also keep low-confidence detections for second association
        # ByteTrack uses low-confidence detections to recover lost tracks
        low_confidence_detections = [
            (bbox, score) for bbox, score in detections 
            if score < self.score_threshold and score >= 0.1  # Very low threshold
        ]
        
        # STEP 2: SEPARATE TRACKS
        # Confirmed tracks: We've seen this box 3+ times (reliable)
        # Unconfirmed tracks: We've seen this box < 3 times (might be false detection)
        confirmed_tracks = [
            track for track in self.tracks 
            if track.hit_streak >= self.min_hits
        ]
        unconfirmed_tracks = [
            track for track in self.tracks 
            if track.hit_streak < self.min_hits
        ]
        
        # STEP 3: FIRST ASSOCIATION - Match high-confidence detections to confirmed tracks
        # This is the main matching step - match detections we're confident about
        matched_pairs, unmatched_detection_indices, unmatched_track_indices = \
            self._match_detections_to_tracks(high_confidence_detections, confirmed_tracks)
        
        # STEP 4: UPDATE MATCHED TRACKS (from first association)
        # For each match, update the track with new position
        for detection_idx, track_idx in matched_pairs:
            bbox, score = high_confidence_detections[detection_idx]
            confirmed_tracks[track_idx].update(bbox, score, self.frame_id)
        
        # STEP 5: SECOND ASSOCIATION - Match remaining detections to unconfirmed tracks
        # This is ByteTrack's key innovation: use low-confidence detections to recover tracks
        unmatched_detections = [
            high_confidence_detections[idx] 
            for idx in unmatched_detection_indices
        ]
        remaining_unmatched_indices = set(unmatched_detection_indices)
        
        if len(unmatched_detections) > 0 and len(unconfirmed_tracks) > 0:
            matched_pairs_2, _, _ = self._match_detections_to_tracks(
                unmatched_detections, unconfirmed_tracks
            )
            
            for detection_idx, track_idx in matched_pairs_2:
                bbox, score = unmatched_detections[detection_idx]
                unconfirmed_tracks[track_idx].update(bbox, score, self.frame_id)
                # Map back to original index
                original_idx = unmatched_detection_indices[detection_idx]
                remaining_unmatched_indices.discard(original_idx)
        
        # STEP 6: CREATE NEW TRACKS
        # Any detection that didn't match any track = new box
        for detection_idx in remaining_unmatched_indices:
            bbox, score = high_confidence_detections[detection_idx]
            new_track = BoxTrack(self.track_id_counter, bbox, score, self.frame_id)
            self.tracks.append(new_track)
            self.track_id_counter += 1  # Give next box a new ID
        
        # STEP 7: REMOVE OLD TRACKS
        # Delete tracks we haven't seen for too long
        active_tracks = []
        for track in self.tracks:
            # Keep track if it was updated this frame
            if track.frame_id == self.frame_id:
                active_tracks.append(track)
            # Or if it's still within max_age (might come back - handles occlusion)
            elif self.frame_id - track.frame_id <= self.max_age:
                active_tracks.append(track)
            # Otherwise, delete it (box left or false detection)
        
        self.tracks = active_tracks
        
        # STEP 8: RETURN ONLY CONFIRMED ACTIVE TRACKS
        # Only return tracks we're confident about and that were updated this frame
        confirmed_active_tracks = [
            track for track in self.tracks 
            if track.hit_streak >= self.min_hits and track.frame_id == self.frame_id
        ]
        return confirmed_active_tracks
    
    def get_track_by_id(self, track_id: int) -> Optional[BoxTrack]:
        """
        STEP 2.5: GET TRACK BY ID
        Find a specific track by its ID number.
        
        Args:
            track_id: The unique ID of the track to find
        
        Returns:
            BoxTrack object if found, None otherwise
        """
        for track in self.tracks:
            if track.track_id == track_id:
                return track
        return None
