"""
STEP 1: OBJECT TRACKING MODULE
================================
This module tracks each person across video frames.
Each person gets a unique ID that stays the same as they move.

FLOW: Detection → Track → Update Track → Match to Next Frame
"""

import numpy as np
from typing import List, Tuple, Optional


class Track:
    """
    STEP 1.1: TRACK CLASS
    ======================
    This represents ONE person being tracked.
    Stores: position, ID, movement history
    """
    
    def __init__(self, track_id: int, bbox: List[float], score: float, frame_id: int):
        """
        STEP 1.1.1: CREATE A NEW TRACK
        When we first detect a person, create a Track object for them.
        """
        # Store the person's unique ID (0, 1, 2, 3, ...)
        self.track_id = track_id
        
        # Bounding box: [left_x, top_y, right_x, bottom_y]
        self.bbox = bbox
        
        # How confident YOLO is that this is a person (0.0 to 1.0)
        self.score = score
        
        # Which frame number we first saw this person
        self.frame_id = frame_id
        
        # Calculate center point of the bounding box
        self.center = self._calculate_center()
        
        # Store all past positions (to see movement direction)
        self.history = [self.center]
        
        # How many frames this track has existed
        self.age = 1
        
        # How many times we've detected this person in a row
        self.hit_streak = 1
        
    def _calculate_center(self) -> Tuple[float, float]:
        """
        STEP 1.1.2: FIND CENTER POINT
        Calculate the center (x, y) of the bounding box.
        This is what we use to track movement.
        """
        x1, y1, x2, y2 = self.bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        return (center_x, center_y)
    
    def update(self, bbox: List[float], score: float, frame_id: int):
        """
        STEP 1.1.3: UPDATE TRACK WITH NEW POSITION
        When we see this person again in the next frame, update their position.
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
        
        # Keep only last 30 positions (to save memory)
        if len(self.history) > 30:
            self.history = self.history[-30:]
    
    def predict(self) -> Tuple[float, float]:
        """
        STEP 1.1.4: PREDICT NEXT POSITION (Optional - for future use)
        Estimate where this person will be in the next frame.
        """
        if len(self.history) < 2:
            return self.center
        
        # Calculate movement speed (velocity)
        last_x, last_y = self.history[-1]
        prev_x, prev_y = self.history[-2]
        dx = last_x - prev_x  # movement in x direction
        dy = last_y - prev_y  # movement in y direction
        
        # Predict next position
        predicted_x = self.center[0] + dx
        predicted_y = self.center[1] + dy
        return (predicted_x, predicted_y)


class SimpleByteTracker:
    """
    STEP 1.2: TRACKER CLASS
    =======================
    This manages ALL tracks (all people being tracked).
    
    MAIN JOB: Match new detections to existing tracks.
    - If detection matches existing track → Update that track
    - If detection is new → Create new track
    - If track not seen for too long → Delete track
    """
    
    def __init__(self, 
                 max_age: int = 30,
                 min_hits: int = 3,
                 iou_threshold: float = 0.3,
                 score_threshold: float = 0.5):
        """
        STEP 1.2.1: INITIALIZE TRACKER
        Set up the tracker with configuration values.
        """
        # If we don't see a person for 30 frames, delete their track
        self.max_age = max_age
        
        # Person must be detected 3 times in a row to be "confirmed"
        self.min_hits = min_hits
        
        # How much boxes must overlap to be considered the same person (0.0 to 1.0)
        self.iou_threshold = iou_threshold
        
        # Minimum confidence score to consider a detection (0.0 to 1.0)
        self.score_threshold = score_threshold
        
        # List of all tracks (all people we're tracking)
        self.tracks: List[Track] = []
        
        # Counter to give each person a unique ID (0, 1, 2, 3, ...)
        self.track_id_counter = 0
        
        # Current frame number
        self.frame_id = 0
        
    def _iou(self, box1: List[float], box2: List[float]) -> float:
        """
        STEP 1.2.2: CALCULATE IoU (Intersection over Union)
        ====================================================
        This measures how much two boxes overlap.
        Returns: 0.0 (no overlap) to 1.0 (perfect overlap)
        
        Example:
        - IoU = 0.9 → Very likely the same person
        - IoU = 0.1 → Probably different people
        """
        # Extract coordinates from both boxes
        x1_box1, y1_box1, x2_box1, y2_box1 = box1
        x1_box2, y1_box2, x2_box2, y2_box2 = box2
        
        # STEP 1: Find the overlapping area (intersection)
        overlap_left = max(x1_box1, x1_box2)
        overlap_top = max(y1_box1, y1_box2)
        overlap_right = min(x2_box1, x2_box2)
        overlap_bottom = min(y2_box1, y2_box2)
        
        # Check if boxes actually overlap
        if overlap_right < overlap_left or overlap_bottom < overlap_top:
            return 0.0  # No overlap
        
        # Calculate intersection area
        intersection_area = (overlap_right - overlap_left) * (overlap_bottom - overlap_top)
        
        # Calculate area of each box
        area_box1 = (x2_box1 - x1_box1) * (y2_box1 - y1_box1)
        area_box2 = (x2_box2 - x1_box2) * (y2_box2 - y1_box2)
        
        # Calculate union (total area covered by both boxes)
        union_area = area_box1 + area_box2 - intersection_area
        
        # Avoid division by zero
        if union_area == 0:
            return 0.0
        
        # IoU = intersection / union
        iou_score = intersection_area / union_area
        return iou_score
    
    def _match_detections_to_tracks(self, 
                                    detections: List[Tuple[List[float], float]],
                                    tracks: List[Track]) -> Tuple[List[int], List[int], List[int]]:
        """
        STEP 1.2.3: MATCH DETECTIONS TO TRACKS
        =======================================
        This is the CORE matching logic.
        
        GOAL: Figure out which detection belongs to which existing track.
        
        PROCESS:
        1. Calculate IoU between every detection and every track
        2. Match the ones with high IoU (they overlap a lot)
        3. Return: matched pairs, unmatched detections, unmatched tracks
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
        # This ensures we match the best overlaps first
        potential_matches.sort(key=lambda x: x[2], reverse=True)
        
        # STEP 4: Greedy matching - match best overlaps first
        matched_pairs = []  # List of (detection_idx, track_idx)
        used_detection_indices = set()
        used_track_indices = set()
        unmatched_track_indices = list(range(len(tracks)))
        
        for detection_idx, track_idx, _ in potential_matches:
            # Only match if neither detection nor track is already used
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
    
    def update(self, detections: List[Tuple[List[float], float]]) -> List[Track]:
        """
        STEP 1.2.4: MAIN UPDATE FUNCTION
        ================================
        This is called for EVERY frame.
        
        PROCESS:
        1. Filter detections (remove low confidence)
        2. Match detections to existing tracks
        3. Update matched tracks
        4. Create new tracks for unmatched detections
        5. Remove old tracks that haven't been seen
        
        Returns: List of confirmed active tracks
        """
        # Move to next frame
        self.frame_id += 1
        
        # STEP 1: FILTER DETECTIONS
        # Only keep detections with confidence above threshold
        high_confidence_detections = [
            (bbox, score) for bbox, score in detections 
            if score >= self.score_threshold
        ]
        
        # STEP 2: SEPARATE TRACKS
        # Confirmed tracks: We've seen this person 3+ times (reliable)
        # Unconfirmed tracks: We've seen this person < 3 times (might be false detection)
        confirmed_tracks = [
            track for track in self.tracks 
            if track.hit_streak >= self.min_hits
        ]
        unconfirmed_tracks = [
            track for track in self.tracks 
            if track.hit_streak < self.min_hits
        ]
        
        # STEP 3: MATCH DETECTIONS TO CONFIRMED TRACKS
        # Try to match new detections to tracks we're confident about
        matched_pairs, unmatched_detection_indices, unmatched_track_indices = \
            self._match_detections_to_tracks(high_confidence_detections, confirmed_tracks)
        
        # STEP 4: UPDATE MATCHED TRACKS
        # For each match, update the track with new position
        for detection_idx, track_idx in matched_pairs:
            bbox, score = high_confidence_detections[detection_idx]
            confirmed_tracks[track_idx].update(bbox, score, self.frame_id)
        
        # STEP 5: TRY TO MATCH UNMATCHED DETECTIONS TO UNCONFIRMED TRACKS
        # Maybe these detections belong to tracks we're not sure about yet
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
        # Any detection that didn't match any track = new person
        for detection_idx in remaining_unmatched_indices:
            bbox, score = high_confidence_detections[detection_idx]
            new_track = Track(self.track_id_counter, bbox, score, self.frame_id)
            self.tracks.append(new_track)
            self.track_id_counter += 1  # Give next person a new ID
        
        # STEP 7: REMOVE OLD TRACKS
        # Delete tracks we haven't seen for too long
        active_tracks = []
        for track in self.tracks:
            # Keep track if it was updated this frame
            if track.frame_id == self.frame_id:
                active_tracks.append(track)
            # Or if it's still within max_age (might come back)
            elif self.frame_id - track.frame_id <= self.max_age:
                active_tracks.append(track)
            # Otherwise, delete it (person left or false detection)
        
        self.tracks = active_tracks
        
        # STEP 8: RETURN ONLY CONFIRMED ACTIVE TRACKS
        # Only return tracks we're confident about and that were updated this frame
        confirmed_active_tracks = [
            track for track in self.tracks 
            if track.hit_streak >= self.min_hits and track.frame_id == self.frame_id
        ]
        return confirmed_active_tracks
    
    def get_track_by_id(self, track_id: int) -> Optional[Track]:
        """
        STEP 1.2.5: GET TRACK BY ID
        Find a specific track by its ID number.
        """
        for track in self.tracks:
            if track.track_id == track_id:
                return track
        return None
