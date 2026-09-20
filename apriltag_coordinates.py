"""The application tag frame, expressed in each backend's original tag frame."""

import cv2
import numpy as np


# Columns are the new unit axes in the original tag frame:
# new +X = old +Z, new +Y = old -X, new +Z = old -Y.
# Points transform as p_old = TAG_BASIS_OLD_FROM_NEW @ p_new.
TAG_BASIS_OLD_FROM_NEW = np.array(
    [[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]],
    dtype=np.float64,
)
TAG_BASIS_OLD_FROM_NEW.setflags(write=False)


def remap_tag_rotation(rotation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (rvec, R_camera_new_tag), keeping the camera frame and origin.

    Apply once after native pose solving/reprojection, before publishing the
    pose. Camera-relative translation is unchanged by this tag basis change.
    """

    remapped = np.asarray(rotation, dtype=np.float64).reshape(3, 3) @ TAG_BASIS_OLD_FROM_NEW
    rvec, _ = cv2.Rodrigues(remapped)
    return rvec, remapped
