import cv2
import numpy as np

class MotionDetector:
    def __init__(self, history=500, var_threshold=50, detect_shadows=True,
                 area_ratio=0.0005, min_contour_area=500):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=detect_shadows
        )
        self.area_ratio = area_ratio
        self.min_contour_area = min_contour_area

    def detect_movement(self, frame, roi=None):
        """
        Detects movement in the given frame or RoI. 
        """
        if roi:
            x, y, w, h = roi
            frame = frame[y:y+h, x:x+w]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        fg_mask = self.bg_subtractor.apply(blurred)
        _, thresh = cv2.threshold(fg_mask, 220, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

        change_ratio = np.count_nonzero(thresh) / float(thresh.size)
        if change_ratio < self.area_ratio:
            return False

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) > self.min_contour_area:
                return True
        return False