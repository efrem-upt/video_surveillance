import cv2
import numpy as np
import logging
import sys
import argparse

logging.basicConfig(
    level=logging.INFO, 
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
)

def smooth_data(history, alpha=0.95):
    """Smooth the data using an exponential moving average."""
    smoothed = [history[0]]  
    for i in range(1, len(history)):
        smoothed.append(alpha * history[i] + (1 - alpha) * smoothed[i - 1])
    return smoothed

def determine_movement(position_history, horizontal_position_history, size_history,
                       moving_away_counter, moving_towards_counter,
                       certainty_frames=25, past_movement_code=0, deadband_threshold_position=0.0001, deadband_threshold_size=0.0001):
    """Decides the movement type (towards camera, away, or none) by averaging changes in position and size."""
    if len(position_history) < 3 or len(size_history) < 3:
        return -1, moving_away_counter, moving_towards_counter, False, False, False

    smoothed_positions = smooth_data(position_history)
    smoothed_horizontal_positions = smooth_data(horizontal_position_history)
    smoothed_sizes = smooth_data(size_history)

    position_changes = np.diff(smoothed_positions)
    horizontal_position_changes = np.diff(smoothed_horizontal_positions)
    size_changes = np.diff(smoothed_sizes)

    avg_position_change = np.mean(position_changes)
    avg_horizontal_position_change = np.mean(horizontal_position_changes)
    avg_size_change = np.mean(size_changes)

    position_in_deadband = abs(avg_position_change) < deadband_threshold_position
    size_in_deadband = abs(avg_size_change) < deadband_threshold_size

    if position_in_deadband and size_in_deadband:
        moving_away_counter, moving_towards_counter = 0, 0
        return 0, moving_away_counter, moving_towards_counter, False, False, False

    weight_position = 0.5
    weight_size = 0.5
    combined_change = (weight_position * avg_position_change) + (weight_size * avg_size_change)

    # If horizontal movement is bigger than vertical movement, consider it lateral
    lateral_movement = abs(avg_horizontal_position_change) > abs(avg_position_change)

    moving_right = lateral_movement and avg_horizontal_position_change > 0
    moving_left = lateral_movement and avg_horizontal_position_change < 0

    if combined_change > 0:
        moving_away_counter = 0
        moving_towards_counter += 1
        if moving_towards_counter >= certainty_frames:
            return 1, moving_away_counter, moving_towards_counter, lateral_movement, moving_right, moving_left
        else:
            return past_movement_code, moving_away_counter, moving_towards_counter, lateral_movement, moving_right, moving_left
    elif combined_change < 0:
        moving_towards_counter = 0
        moving_away_counter += 1
        if moving_away_counter >= certainty_frames:
            return 3, moving_away_counter, moving_towards_counter, lateral_movement, moving_right, moving_left
        else:
            return past_movement_code, moving_away_counter, moving_towards_counter, lateral_movement, moving_right, moving_left
    else:
        moving_away_counter, moving_towards_counter = 0, 0
        return 0, moving_away_counter, moving_towards_counter, lateral_movement, moving_right, moving_left

def parse_arguments():
    parser = argparse.ArgumentParser(description="Motion Detection with Tracking")
    parser.add_argument("video_path", type=str, help="Path to the input video file.")
    parser.add_argument("--roi", "-r", action="store_true", help="Enable Region of Interest (RoI) selection.")
    parser.add_argument("--alert", action="store_true", help="Enable door alert feature.")
    parser.add_argument("--heatmap", action="store_true", help="Generate heatmap.")
    parser.add_argument("--live", action="store_true", help="Display heatmap in real-time on video.")
    parser.add_argument("--push", action="store_true", help="Send push notification for door alert.")
    parser.add_argument("--door_wait_threshold", type=int, default=2, help="Number of seconds to wait before considering someone waiting at the door.")
    parser.add_argument("--push_wait_threshold", type=int, default=30, help="Number of seconds to wait before sending another push notification.")
    return parser.parse_args()

def select_roi(frame):
    """
    Lets the user select an RoI from the given frame and returns (x, y, w, h).
    """
    roi = cv2.selectROI("Select RoI", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select RoI")
    if sum(roi) == 0:
        logging.error("No RoI selected. Exiting.")
        sys.exit(1)
    logging.info(f"Selected RoI: x={roi[0]}, y={roi[1]}, w={roi[2]}, h={roi[3]}")
    return roi

def letterbox_resize(image, target_size=(640, 640), color=(114, 114, 114)):
    """
    Resizes an image (while keeping aspect ratio) to target_size using 'letterbox' style padding.
    """
    src_h, src_w = image.shape[:2]
    target_w, target_h = target_size

    scale = min(target_w / src_w, target_h / src_h)
    new_w, new_h = int(src_w * scale), int(src_h * scale)

    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_w = target_w - new_w
    pad_h = target_h - new_h
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    padded_image = cv2.copyMakeBorder(
        resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return padded_image, scale, (left, top)

def draw_roi_box(frame, roi, color=(0, 255, 0), thickness=2, label="RoI"):
    """Draws a labeled bounding box for the selected RoI on the frame."""
    x, y, w, h = roi
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
    (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x, y - label_height - baseline), (x + label_width, y), color, -1)
    cv2.putText(frame, label, (x, y - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)