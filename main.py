import cv2
import numpy as np
import logging
import sys
import os
from ultralytics import YOLO
import time
import json
import argparse

from motion_detection import MotionDetector
from utils import *
from tracking import Tracker
from notifications import send_push_notification
from constants import *

def main(video_path, use_roi, alert_enabled=False, heatmap_enabled=False, live_enabled=False, push=False,
         alert_wait_threshold=2, push_wait_threshold=30):

    global frame_size, config

    door_config = config["door_detection"]
    tracker_config = config["tracker"]

    model = YOLO("yolov8s.pt") if not config['camera_mode'] else YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Could not open video file '{video_path}'.")
        sys.exit(1)

    ret, frame = cap.read()
    if not ret:
        logging.error("Failed to read the first frame.")
        sys.exit(1)

    if config['camera_mode']:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        frame_width = 640
        frame_height = 480
        frame_size = (frame_height, frame_width)
    else:
        frame_size = frame.shape[:2]
        frame_width = frame.shape[1]
        frame_height = frame.shape[0]

    if use_roi:
        if config['save_motion_video']:
                motion_video_roi_dir = "Motion Video - RoI mode"
                if not os.path.exists(motion_video_roi_dir) and config["save_motion_video"]:
                    os.makedirs(motion_video_roi_dir)
                if video_path == 0:
                        output_file = os.path.join(motion_video_roi_dir, "camera_motion_roimode.avi")
                        if os.path.exists(output_file):
                            i = 1
                            while True:
                                new_filename = f"camera_motion_roimode_{i}.avi"
                                new_filename = os.path.join(motion_video_roi_dir, new_filename)
                                if not os.path.exists(new_filename):
                                    output_file = new_filename
                                    break
                                i += 1
                else:
                    output_file = os.path.join(motion_video_roi_dir, os.path.splitext(os.path.basename(video_path))[0] + "_motion_roimode.avi")
    else:
        if config['save_motion_video']:
            motion_video_dir = "Motion Video"
            if not os.path.exists(motion_video_dir):
                os.makedirs(motion_video_dir)
            if video_path == 0:
                    output_file = os.path.join(motion_video_dir, "camera_motion.avi")
                    if os.path.exists(output_file):
                        i = 1
                        while True:
                            new_filename = f"camera_motion_{i}.avi"
                            new_filename = os.path.join(motion_video_dir, new_filename)
                            if not os.path.exists(new_filename):
                                output_file = new_filename
                                break
                            i += 1
            else:
                output_file = os.path.join(motion_video_dir, os.path.splitext(os.path.basename(video_path))[0] + "_motion.avi")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = cap.get(cv2.CAP_PROP_FPS)
    if config["save_motion_video"]:
        video_writer = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

    motion_detector = MotionDetector(history=config['motion_detector']['history'], var_threshold=config['motion_detector']['var_threshold'], detect_shadows=config['motion_detector']['detect_shadows'], area_ratio=config['motion_detector']['area_ratio'], min_contour_area=config['motion_detector']['min_countour_area'])
    frame_skip = 3 if not config['camera_mode'] else 1
    frame_count = 0
    wait_alert_time = 0
    wait_push_again = 0
    ALERT_DOOR_THRESHOLD = alert_wait_threshold
    WAIT_SEND_PUSH_AGAIN = push_wait_threshold
    alert_sent = False
    roi = None

    tracker = Tracker(
        iou_threshold=tracker_config["iou_threshold"],
        max_missed=tracker_config["max_missed"],
        feature_threshold=tracker_config["feature_threshold"],
        reid_threshold=tracker_config["reid_threshold"],
        max_reid=tracker_config["max_reid"],
        history=config['movement_classification']['history'],
        iou_weight=tracker_config["iou_weight"],
        feature_weight=tracker_config["feature_weight"]
    )

    heatmap_dir = "Heatmaps"
    if heatmap_enabled and not os.path.exists(heatmap_dir):
        os.makedirs(heatmap_dir)
    if heatmap_enabled:
        heatmap = np.zeros((frame_size[0], frame_size[1]), dtype=np.float32)
        if video_path == 0:
            heatmap_filename = "heatmap_camera.png"
            heatmap_filename = os.path.join(heatmap_dir, heatmap_filename)
        else:
            video_basename = os.path.splitext(os.path.basename(video_path))[0]
            heatmap_filename = f"{video_basename}_heatmap.png"
            heatmap_filename = os.path.join(heatmap_dir, heatmap_filename)
            if os.path.exists(heatmap_filename):
                i = 1
                while True:
                    new_filename = f"{video_basename}_heatmap_{i}.png"
                    new_filename = os.path.join(heatmap_dir, new_filename)
                    if not os.path.exists(new_filename):
                        heatmap_filename = new_filename
                        break
                    i += 1

    if use_roi:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, first_frame = cap.read()
        if not ret:
            logging.error("Cannot read the first frame for RoI selection.")
            sys.exit(1)

        logging.info("Please select the Region of Interest (RoI) and press ENTER or SPACE when done.")
        selected_roi = select_roi(first_frame)
        x, y, w, h = selected_roi
        roi = (x, y, w, h)
        if config['save_motion_video']:
            video_writer = cv2.VideoWriter(output_file, fourcc, fps, (w, h))
        logging.info(f"Mapped RoI to original frame size: {roi}")
    else:
        logging.info("RoI selection not enabled. Using entire frame for motion detection.")

    if use_roi:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        ret, current_frame = cap.read()
        if not ret:
            logging.info("End of video stream.")
            break

        frame_before_mod = current_frame.copy()
        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        if use_roi and roi:
            draw_roi_box(current_frame, roi, color=(0, 255, 0), thickness=2, label="RoI")

        motion = motion_detector.detect_movement(current_frame, roi=roi) if use_roi else motion_detector.detect_movement(current_frame)

        if motion:
            if use_roi and roi is not None:
                x, y, w, h = roi
                roi_frame = current_frame[y:y+h, x:x+w]
                gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                fg_mask = motion_detector.bg_subtractor.apply(blurred)
                _, thresh = cv2.threshold(fg_mask, 220, 255, cv2.THRESH_BINARY)

                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                boxes = []
                for cnt in contours:
                    if cv2.contourArea(cnt) > motion_detector.min_contour_area:
                        bx, by, bw, bh = cv2.boundingRect(cnt)
                        boxes.append([bx, by, bw, bh])

                detections_roi_dir = "Detections - RoI mode"
                if not os.path.exists(detections_roi_dir) and config["save_detections"]:
                    os.makedirs(detections_roi_dir)

                if boxes:
                    indices = cv2.dnn.NMSBoxes(boxes, [1.0]*len(boxes), score_threshold=0.3, nms_threshold=0.35)
                    for i in indices:
                        idx = i[0] if isinstance(i, (list, tuple, np.ndarray)) else i
                        bx, by, bw, bh = boxes[idx]
                        cv2.rectangle(current_frame, (x+bx, y+by), (x+bx+bw, y+by+bh), (0,255,255), 2)
                        crop_img = frame_before_mod[y+by:y+by+bh, x+bx:x+bx+bw]
                        if config["save_detections"]:
                            timestamp = time.strftime("%Y%m%d_%H%M%S")
                            filename = os.path.join(detections_roi_dir, f"detection_{timestamp}.jpg")
                            cv2.imwrite(filename, crop_img)
                if config["save_motion_video"]:
                    video_writer.write(roi_frame)
            else:
                resized_frame = cv2.resize(current_frame, (640, 360)) if not config['camera_mode'] else current_frame
                results = model(resized_frame)

                detections = []
                for result in results:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    scores = result.boxes.conf.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy()

                    for box, score, cls in zip(boxes, scores, classes):
                        if int(cls) == 0:  # Person
                            x1, y1, x2, y2 = box
                            x1 = int(x1 * current_frame.shape[1] / resized_frame.shape[1])
                            y1 = int(y1 * current_frame.shape[0] / resized_frame.shape[0])
                            x2 = int(x2 * current_frame.shape[1] / resized_frame.shape[1])
                            y2 = int(y2 * current_frame.shape[0] / resized_frame.shape[0])
                            detections.append((x1, y1, x2, y2))

                if len(detections) > 1:
                    boxes_arr = np.array([det[:4] for det in detections])
                    scores_arr = np.array([1.0] * len(detections))
                    indices = cv2.dnn.NMSBoxes(boxes_arr.tolist(), scores_arr.tolist(),
                                              score_threshold=0.3, nms_threshold=0.35)
                    detections = [detections[i] for i in indices] if indices is not None else detections

                tracker.update(current_frame, detections, frame_size)
                tracks = tracker.get_tracks()

                if heatmap_enabled:
                    for track in tracks:
                        x1, y1, x2, y2 = track.bbox
                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)
                        temp = np.zeros_like(heatmap)
                        cv2.circle(temp, (cx, cy), radius=25, color=1, thickness=-1)
                        heatmap += temp

                for track in tracks:
                    if track.missed > 0:
                        continue
                    movement_code, track.moving_away_counter, track.moving_towards_counter, \
                    track.moving_laterally, track.moving_right, track.moving_left = determine_movement(
                        track.position_history,
                        track.horizontal_position_history,
                        track.size_history,
                        track.moving_away_counter,
                        track.moving_towards_counter,
                        certainty_frames=config['movement_classification']['certainty_frames'],
                        past_movement_code=track.past_movement_code,
                        deadband_threshold_position=config['movement_classification']['deadband_threshold_position'],
                        deadband_threshold_size=config['movement_classification']['deadband_threshold_size']
                    )
                    track.past_movement_code = movement_code

                    if alert_enabled:
                        x1, y1, x2, y2 = track.bbox
                        box_width = x2 - x1
                        if (box_width > door_config["width_ratio"] * frame_size[1] and
                            y2 > door_config["y_position_ratio"] * frame_size[0]):
                            wait_alert_time += 1
                            if (wait_alert_time > ALERT_DOOR_THRESHOLD * fps / frame_skip) and not alert_sent:
                                alert_sent = True
                                text = "ALERT: Person at the door!"
                                font = cv2.FONT_HERSHEY_SIMPLEX
                                font_scale = 0.7
                                thickness = 2

                                (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                                padding = 20
                                x_alert = frame_size[1] - text_w - padding
                                y_alert = text_h + padding

                                overlay = current_frame.copy()
                                cv2.rectangle(overlay,
                                              (x_alert - 10, y_alert - text_h - 10),
                                              (x_alert + text_w + 10, y_alert + baseline + 10),
                                              (0, 0, 0), -1)
                                alpha = 0.6
                                cv2.addWeighted(overlay, alpha, current_frame, 1 - alpha, 0, current_frame)

                                cv2.putText(current_frame, text, (x_alert, y_alert), font, font_scale,
                                            (255, 255, 255), thickness + 2)
                                cv2.putText(current_frame, text, (x_alert, y_alert), font, font_scale,
                                            (0, 0, 255), thickness)
                                time_start_text = time.time()

                                if push:
                                    send_push_notification("Someone is at the door!")
                            elif alert_sent:
                                time_end_text = time.time()
                                if time_end_text - time_start_text < door_config["display_duration_sec"]:
                                    text = "ALERT: Person at the door!"
                                    font = cv2.FONT_HERSHEY_SIMPLEX
                                    font_scale = 0.7
                                    thickness = 2

                                    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                                    padding = 20
                                    x_alert = frame_size[1] - text_w - padding
                                    y_alert = text_h + padding

                                    overlay = current_frame.copy()
                                    cv2.rectangle(overlay,
                                                  (x_alert - 10, y_alert - text_h - 10),
                                                  (x_alert + text_w + 10, y_alert + baseline + 10),
                                                  (0, 0, 0), -1)
                                    alpha = 0.6
                                    cv2.addWeighted(overlay, alpha, current_frame, 1 - alpha, 0, current_frame)

                                    cv2.putText(current_frame, text, (x_alert, y_alert), font, font_scale,
                                                (255, 255, 255), thickness + 2)
                                    cv2.putText(current_frame, text, (x_alert, y_alert), font, font_scale,
                                                (0, 0, 255), thickness)

                                wait_push_again += 1
                                if wait_push_again > WAIT_SEND_PUSH_AGAIN * fps / frame_skip:
                                    alert_sent = False
                                    wait_push_again = 0
                                wait_alert_time = 0

                    color = (255, 255, 255)
                    if movement_code == MOVEMENT_MOVING_TOWARDS_CAMERA:
                        color = (0, 255, 0)
                    elif movement_code == MOVEMENT_MOVING_AWAY_CAMERA:
                        color = (0, 0, 255)
                    elif movement_code == MOVEMENT_UNKNOWN_MOVEMENT:
                        color = (0, 255, 255)
                    else:
                        color = (255, 255, 255)

                    if track.missed == 0:
                        x1, y1, x2, y2 = track.bbox
                        cv2.rectangle(current_frame, (x1, y1), (x2, y2), color, 2)
                        movement_label = {
                            MOVEMENT_NO_MOVEMENT: "No Movement",
                            MOVEMENT_MOVING_TOWARDS_CAMERA: "Moving Towards",
                            MOVEMENT_MOVING_AWAY_CAMERA: "Moving Away",
                        }.get(movement_code, "Unknown")

                        if track.moving_laterally:
                            movement_label += " (Lateral)"
                        if track.moving_right:
                            movement_label += " (Right)"
                        elif track.moving_left:
                            movement_label += " (Left)"

                        label = f"Person {track.id}: {movement_label}"
                        cv2.putText(current_frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                if config["save_motion_video"]:
                    video_writer.write(current_frame)

                detections_dir = "Detections"
                if not os.path.exists(detections_dir):
                    os.makedirs(detections_dir)
                for track in tracks:
                    if track.missed == 0:
                        x1, y1, x2, y2 = track.bbox
                        crop_img = frame_before_mod[y1:y2, x1:x2]
                        if config["save_detections"]:
                            timestamp = time.strftime("%Y%m%d_%H%M%S")
                            filename = os.path.join(detections_dir, f"person_{track.id}_{timestamp}.jpg")
                            cv2.imwrite(filename, crop_img)
        else:
            if use_roi and roi:
                x, y, w, h = roi
                cv2.line(current_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.line(current_frame, (x + w, y), (x, y + h), (0, 0, 255), 2)
                cv2.putText(current_frame, "Removed", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                h_frame, w_frame, _ = current_frame.shape
                cv2.line(current_frame, (0, 0), (w_frame, h_frame), (0, 0, 255), 2)
                cv2.line(current_frame, (w_frame, 0), (0, h_frame), (0, 0, 255), 2)
                cv2.putText(current_frame, "Removed", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Live heatmap overlay 
        if heatmap_enabled and live_enabled and not use_roi:
            heatmap_blurred_live = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=15, sigmaY=15)
            heatmap_uint8_live = cv2.normalize(heatmap_blurred_live, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            colored_heatmap_live = cv2.applyColorMap(heatmap_uint8_live, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(current_frame, 0.7, colored_heatmap_live, 0.3, 0)
            cv2.imshow("FCV - Project 2 (Efrem Dragos-Sebastian-Mihaly)", overlay)
        else:
            cv2.imshow("FCV - Project 2 (Efrem Dragos-Sebastian-Mihaly)", current_frame)

        if cv2.waitKey(30) & 0xFF == 27:
            logging.info("Escape key pressed. Exiting.")
            break

    cap.release()
    if config["save_motion_video"]:
        video_writer.release()
    cv2.destroyAllWindows()

    if heatmap_enabled and not use_roi:
        heatmap_blurred = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=15, sigmaY=15)
        heatmap_uint8 = cv2.normalize(heatmap_blurred, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        cv2.imwrite(heatmap_filename, colored_heatmap)
        logging.info(f"Heatmap saved as {heatmap_filename}")

    logging.info("Motion Detection and Tracking terminated.")


if __name__ == "__main__":

    default_config = {
        "video_path": "input.mp4",
        "camera_mode": True,
        "roi_mode": False,
        "save_detections": False,
        "save_motion_video": False,
        "motion_detector": {
             "history": 500,
            "var_threshold": 50,
            "detect_shadows": True,
            "area_ratio": 0.0005,
            "min_countour_area": 500
        },
        "tracker": {
            "iou_threshold": 0.1,
            "max_missed": 100,
            "feature_threshold": 0.1,
            "reid_threshold": 0.2,
            "max_reid": 50,
            "iou_weight": 0.5,
            "feature_weight": 0.5
        },
         "movement_classification": {
            "history": 15,
            "certainty_frames": 10,
            "deadband_threshold_position": 0.0001,
            "deadband_threshold_size": 0.0001
        },
        "door_detection": {
            "alert": False,
            "width_ratio": 0.142857,
            "y_position_ratio": 0.95,
            "display_duration_sec": 5,
            "door_wait_threshold": 2,
            "push_wait_threshold": 30,
            "push": False
        },
        "heatmap": {
            "active": False,
            "live": False
        }
    }

    parser = argparse.ArgumentParser(description="Motion Detection with Tracking")
    parser.add_argument("--config", type=str, help="Path to the config file.", required=True)
    args = parser.parse_args()

    try:
        with open(args.config, "r") as f:
            user_config = json.load(f)
    except Exception as e:
        print(f"Error loading config.json: {e}")
        user_config = {}

    config = {**default_config, **user_config}

    if config["camera_mode"]:
        config["video_path"] = 0

    cap = cv2.VideoCapture(config["video_path"])
    _, frame = cap.read()
    if frame is not None:
        frame_size = frame.shape[:2]
    cap.release()

    fps = cv2.VideoCapture(config["video_path"]).get(cv2.CAP_PROP_FPS)

    print(config)

    main(
        config["video_path"],
        config["roi_mode"],
        alert_enabled=config["door_detection"]["alert"],
        heatmap_enabled=config["heatmap"]["active"],
        live_enabled=config["heatmap"]["live"],
        push=config["door_detection"]["push"],
        alert_wait_threshold=config["door_detection"]["door_wait_threshold"],
        push_wait_threshold=config["door_detection"]["push_wait_threshold"]
    )