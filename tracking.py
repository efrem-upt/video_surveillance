from itertools import count
from collections import deque
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from constants import *
import logging

class Track:
    _ids = count(0)

    def __init__(self, bbox, feature, frame_size, history):
        self.id = next(self._ids)
        self.bbox = bbox  # (x1, y1, x2, y2)
        self.feature = feature  # Appearance feature vector
        self.age = 0
        self.missed = 0
        self.history = [bbox]
        self.confidence = 1.0
        self.frame_size = frame_size

        # For movement detection
        self.position_history = deque(maxlen=history)
        self.horizontal_position_history = deque(maxlen=history)
        self.size_history = deque(maxlen=history)
        self.moving_away_counter = 0
        self.moving_towards_counter = 0
        self.moving_laterally = False
        self.moving_right = False
        self.moving_left = False
        self.past_movement_code = MOVEMENT_NO_MOVEMENT

        self._update_histories(bbox)

    def _update_histories(self, bbox):
        """
        Updates the position/size history relative to the entire frame_size.
        """
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = y2
        size = y2 - y1
        # Normalize by entire frame's height/width
        self.position_history.append(center_y / self.frame_size[0])
        self.horizontal_position_history.append(center_x / self.frame_size[1])
        self.size_history.append(size / self.frame_size[0])

    def update(self, bbox, feature, score=1.0):
        """
        Called to update the track with a new bounding box, feature, and confidence score.
        """
        self.bbox = bbox
        self.feature = feature
        self.age += 1
        self.missed = 0
        self.history.append(bbox)
        if len(self.history) > 2:
            self.history.pop(0)
        self.confidence = score
        self._update_histories(bbox)

    def mark_missed(self):
        """Marks a track as missed (no matching detection in this frame)."""
        self.missed += 1
        self.age += 1
        self.confidence *= 0.9


class Tracker:
    def __init__(self, iou_threshold=0.3, max_missed=50,
                 feature_threshold=0.5, reid_threshold=0.7, max_reid=50, history=15, iou_weight=0.5, feature_weight=0.5):
        self.tracks = []
        self.reid_pool = []
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.feature_threshold = feature_threshold
        self.reid_threshold = reid_threshold
        self.max_reid = max_reid
        self.history = history
        self.iou_weight = iou_weight
        self.feature_weight = feature_weight

    def iou(self, bbox1, bbox2):
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)

        inter_width = max(0, xi2 - xi1)
        inter_height = max(0, yi2 - yi1)
        inter_area = inter_width * inter_height

        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - inter_area

        if union_area == 0:
            return 0
        return inter_area / union_area

    def extract_color_histogram(self, frame, bbox):
        """
        Extracts an HSV color histogram from the bounding box region
        of the frame (used for re-ID / feature similarity).
        """
        x1, y1, x2, y2 = bbox
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return np.zeros((300,), dtype=np.float32)
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv_roi], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten()

    def feature_similarity(self, feature1, feature2):
        if feature1.size == 0 or feature2.size == 0:
            return 0
        return cv2.compareHist(feature1.astype('float32'),
                               feature2.astype('float32'),
                               cv2.HISTCMP_CORREL)

    def update(self, frame, detections, frame_size):
        """
        Updates all existing tracks with the new detections, creates new tracks for unmatched detections,
        and handles re-identification of tracks that were lost.
        """
        assignments = [-1] * len(detections)

        if len(self.tracks) == 0:
            for det in detections:
                feature = self.extract_color_histogram(frame, det)
                self.tracks.append(Track(det, feature, frame_size, history=self.history))
            return

        if len(detections) == 0:
            for track in self.tracks:
                track.mark_missed()
            self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]
            return

        iou_matrix = np.zeros((len(self.tracks), len(detections)), dtype=np.float32)
        for t, track in enumerate(self.tracks):
            for d, det in enumerate(detections):
                iou_matrix[t, d] = self.iou(track.bbox, det)

        feature_matrix = np.zeros((len(self.tracks), len(detections)), dtype=np.float32)
        for t, track in enumerate(self.tracks):
            for d, det in enumerate(detections):
                det_feature = self.extract_color_histogram(frame, det)
                feature_matrix[t, d] = self.feature_similarity(track.feature, det_feature)

        if np.max(iou_matrix) > 0:
            iou_matrix = iou_matrix / np.max(iou_matrix)
        if np.max(feature_matrix) > 0:
            feature_matrix = feature_matrix / np.max(feature_matrix)

        # Combine IoU, feature similarity matrices
        weights = [self.iou_weight, self.feature_weight]
        combined_matrix = (weights[0]*iou_matrix +
                           weights[1]*feature_matrix)

        cost_matrix = 1 - combined_matrix

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < (1 - self.iou_threshold) and feature_matrix[r, c] > self.feature_threshold:
                assignments[c] = r

        assigned_tracks = set()
        for d, t in enumerate(assignments):
            if t != -1:
                new_feature = self.extract_color_histogram(frame, detections[d])
                self.tracks[t].update(detections[d], new_feature)
                assigned_tracks.add(t)

        # Re-ID logic for unmatched detections
        unmatched_detections = [d for d, t in enumerate(assignments) if t == -1]
        for d in unmatched_detections:
            det_feature = self.extract_color_histogram(frame, detections[d])
            best_match = -1
            best_similarity = -1
            # Check if it matches a track in the reid_pool
            for idx, terminated_track in enumerate(self.reid_pool):
                similarity = self.feature_similarity(terminated_track.feature, det_feature)
                if similarity > self.reid_threshold and similarity > best_similarity:
                    best_similarity = similarity
                    best_match = idx
            if best_match != -1:
                # Re-identify an old track
                reassigned_track = self.reid_pool.pop(best_match)
                reassigned_track.update(detections[d], det_feature)
                self.tracks.append(reassigned_track)
                logging.info(f"Re-identifying Track ID {reassigned_track.id} with new detection.")
            else:
                self.tracks.append(Track(detections[d], det_feature, frame_size, history=self.history))

        # Mark unassigned tracks as missed
        for t, track in enumerate(self.tracks):
            if t not in assigned_tracks:
                track.mark_missed()

        # Clean up old / missed tracks
        remaining_tracks = []
        for track in self.tracks:
            if track.missed > self.max_missed:
                # Keep it for possible re-ID
                self.reid_pool.append(track)
            else:
                remaining_tracks.append(track)
        self.tracks = remaining_tracks

        # Limit the size of the re-ID pool
        while len(self.reid_pool) > self.max_reid:
            self.reid_pool.pop(0)

    def get_tracks(self):
        return self.tracks