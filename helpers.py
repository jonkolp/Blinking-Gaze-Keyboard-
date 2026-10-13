"""
Helper functions and constants for eye gaze and blink detection.

Provides coordinate conversion utilities, MediaPipe landmark indices,
and Eye Aspect Ratio (EAR) computation for blink classification.
"""

from typing import List, Tuple
import numpy as np


# ============================================================================
# MediaPipe FaceMesh Landmark Indices
# ============================================================================

# 6 landmarks for EAR computation per eye (Soukupova & Cech, 2016)
LEFT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]

# Refined iris center landmarks (available with refine_landmarks=True)
LEFT_PUPIL_LANDMARK = 468
RIGHT_PUPIL_LANDMARK = 473

# 6 landmarks used for 3D Head Pose estimation (solvePnP)
# Nose tip, Chin, Left eye corner, Right eye corner, Left mouth corner, Right mouth corner
HEAD_POSE_LANDMARKS = [4, 152, 263, 33, 287, 57]


# ============================================================================
# Coordinate Conversion Utilities
# ============================================================================

def relative(landmark, shape: Tuple[int, int, ...]) -> Tuple[int, int]:
    """
    Convert a normalized MediaPipe landmark (0.0 to 1.0) into 2D pixel coordinates.

    :param landmark: Normalized landmark with .x and .y attributes.
    :param shape: Image shape (height, width, ...).
    :return: (x, y) pixel coordinates as integers.
    """
    return int(landmark.x * shape[1]), int(landmark.y * shape[0])


def relativeT(landmark, shape: Tuple[int, int, ...]) -> Tuple[int, int, int]:
    """
    Convert a normalized MediaPipe landmark into (x, y, 0) coordinates for 3D estimation.

    :param landmark: Normalized landmark with .x and .y attributes.
    :param shape: Image shape (height, width, ...).
    :return: (x, y, 0) coordinates.
    """
    return int(landmark.x * shape[1]), int(landmark.y * shape[0]), 0


# ============================================================================
# Blink Detection - Eye Aspect Ratio (EAR)
# ============================================================================

def eye_aspect_ratio(landmarks, frame_shape: Tuple[int, int, ...]) -> float:
    """
    Calculate the Eye Aspect Ratio (EAR) for blink detection.

    Formula (Soukupova & Cech, 2016):
        EAR = (||p2 - p6|| + ||p3 - p5||) / (2.0 * ||p1 - p4||)

    :param landmarks: Ordered list of 6 landmark points corresponding to the eye.
    :param frame_shape: Shape of the frame (height, width, ...).
    :return: EAR floating-point value. Lower values indicate eye closure.
    """
    if len(landmarks) < 6:
        return 0.0

    # Convert landmarks to pixel coordinates
    points = [np.array(relative(lm, frame_shape), dtype=np.float64) for lm in landmarks[:6]]
    p1, p2, p3, p4, p5, p6 = points

    # Compute Euclidean vertical distances
    vertical_1 = np.linalg.norm(p2 - p6)
    vertical_2 = np.linalg.norm(p3 - p5)

    # Compute Euclidean horizontal distance
    horizontal = np.linalg.norm(p1 - p4)

    if horizontal == 0:
        return 0.0

    # Calculate and return EAR
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return float(ear)
