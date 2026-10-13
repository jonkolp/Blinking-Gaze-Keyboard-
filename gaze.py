"""
Gaze estimation module using MediaPipe FaceMesh and OpenCV.

Computes 3D head pose via solvePnP, estimates eye pupil coordinates in 3D,
and projects the resulting gaze ray back onto the camera frame.
"""

from typing import Optional, Tuple
import cv2
import numpy as np
from helpers import (
    HEAD_POSE_LANDMARKS,
    LEFT_PUPIL_LANDMARK,
    RIGHT_PUPIL_LANDMARK,
    relative,
    relativeT,
)


# Standard 3D anthropometric face model coordinates (in mm)
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),          # Landmark 4: Nose tip
    (0.0, -63.6, -12.5),      # Landmark 152: Chin
    (-43.3, 32.7, -26.0),     # Landmark 263: Left eye outer corner
    (43.3, 32.7, -26.0),      # Landmark 33: Right eye outer corner
    (-28.9, -28.9, -24.1),    # Landmark 287: Left mouth corner
    (28.9, -28.9, -24.1)      # Landmark 57: Right mouth corner
], dtype=np.float64)

# Canonical 3D eyeball center positions in model space
EYE_BALL_CENTER_LEFT = np.array([[29.05], [32.7], [-39.5]], dtype=np.float64)
EYE_BALL_CENTER_RIGHT = np.array([[-29.05], [32.7], [-39.5]], dtype=np.float64)


class GazeSmoother:
    """Exponential Moving Average (EMA) smoother for 2D gaze coordinates."""

    def __init__(self, alpha: float = 0.35):
        """
        :param alpha: Smoothing factor between 0.0 (maximum smoothing) and 1.0 (no smoothing).
        """
        self.alpha = max(0.0, min(1.0, alpha))
        self.smoothed_point: Optional[Tuple[float, float]] = None

    def update(self, point: Optional[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        """Update filter with a new gaze point and return smoothed integer coordinates."""
        if point is None:
            return None
        if self.smoothed_point is None:
            self.smoothed_point = (float(point[0]), float(point[1]))
        else:
            prev_x, prev_y = self.smoothed_point
            new_x = self.alpha * point[0] + (1.0 - self.alpha) * prev_x
            new_y = self.alpha * point[1] + (1.0 - self.alpha) * prev_y
            self.smoothed_point = (new_x, new_y)

        return int(round(self.smoothed_point[0])), int(round(self.smoothed_point[1]))

    def reset(self):
        """Reset internal filter state."""
        self.smoothed_point = None


def estimate_gaze(frame: np.ndarray, landmarks) -> Optional[Tuple[int, int]]:
    """
    Estimate the 2D gaze point from facial landmarks and draw the gaze ray.

    :param frame: BGR input image frame from OpenCV.
    :param landmarks: MediaPipe FaceMesh landmarks object.
    :return: (x, y) 2D gaze point projected onto camera frame, or None if estimation fails.
    """
    h, w = frame.shape[:2]

    # Extract 2D image coordinates for head pose estimation
    image_points_2d = np.array(
        [relative(landmarks.landmark[idx], frame.shape) for idx in HEAD_POSE_LANDMARKS],
        dtype=np.float64
    )

    # 3D points with z=0 for affine transformation
    image_points_3d = np.array(
        [relativeT(landmarks.landmark[idx], frame.shape) for idx in HEAD_POSE_LANDMARKS],
        dtype=np.float64
    )

    # Approximate intrinsic camera matrix (assuming principal point at center, no distortion)
    focal_length = float(w)
    center = (w / 2.0, h / 2.0)
    camera_matrix = np.array([
        [focal_length, 0.0, center[0]],
        [0.0, focal_length, center[1]],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    # Solve Perspective-n-Point to find head orientation and translation
    success, rotation_vector, translation_vector = cv2.solvePnP(
        MODEL_POINTS, image_points_2d, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return None

    # Extract 2D pupil coordinates
    left_pupil = relative(landmarks.landmark[LEFT_PUPIL_LANDMARK], frame.shape)

    # Estimate affine transformation from 2D facial plane to 3D head model
    _, transformation, _ = cv2.estimateAffine3D(image_points_3d, MODEL_POINTS)
    if transformation is None:
        return None

    # Map left pupil coordinate to 3D world/model coordinate
    pupil_homogeneous = np.array([[left_pupil[0], left_pupil[1], 0.0, 1.0]], dtype=np.float64).T
    pupil_world_cord = transformation @ pupil_homogeneous

    # Project gaze ray from eyeball center outwards
    gaze_ray_3d = EYE_BALL_CENTER_LEFT + (pupil_world_cord - EYE_BALL_CENTER_LEFT) * 10.0

    # Project 3D gaze point back into 2D camera coordinates
    projected_points, _ = cv2.projectPoints(
        gaze_ray_3d.T, rotation_vector, translation_vector, camera_matrix, dist_coeffs
    )

    gaze_point = (int(projected_points[0][0][0]), int(projected_points[0][0][1]))

    # Draw gaze vector ray on frame: from pupil to projected gaze point
    cv2.line(frame, (int(left_pupil[0]), int(left_pupil[1])), gaze_point, (0, 0, 255), 2)
    cv2.circle(frame, (int(left_pupil[0]), int(left_pupil[1])), 3, (0, 255, 0), -1)

    return gaze_point


# Backward-compatible alias for existing imports
gaze = estimate_gaze
