import math
from typing import Sequence

import numpy as np


TIP_LANDMARKS = [4, 8, 12, 16, 20]


def normalize_landmarks(landmarks: Sequence[object]) -> np.ndarray:
    """Return the 47-feature hand vector used by the Random Forest model."""
    wrist_x = landmarks[0].x
    wrist_y = landmarks[0].y
    wrist_z = landmarks[0].z

    pts = np.array([(lm.x - wrist_x, lm.y - wrist_y) for lm in landmarks], dtype=np.float32)
    zs = np.array([lm.z - wrist_z for lm in landmarks], dtype=np.float32)

    base_x, base_y = pts[9]
    angle = math.atan2(base_y, base_x)
    cos_a = math.cos(-angle)
    sin_a = math.sin(-angle)
    rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)

    pts = pts @ rotation.T
    scale = np.sqrt(np.mean(pts[:, 0] ** 2 + pts[:, 1] ** 2))
    if scale < 1e-6:
        scale = 1e-6

    pts /= scale
    depth_features = (zs[TIP_LANDMARKS] / scale).astype(np.float32)
    return np.concatenate([pts.flatten(), depth_features])
