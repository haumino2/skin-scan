"""Face mesh detection using MediaPipe and region mask generation."""
import cv2
import numpy as np
from typing import Optional

import mediapipe as mp

try:
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    NEW_API = True
except ImportError:
    NEW_API = False


class FaceMeshDetector:
    """MediaPipe face mesh detector."""

    def __init__(self):
        if NEW_API:
            try:
                import urllib.request, os
                model_path = "/tmp/face_landmarker.task"
                if not os.path.exists(model_path):
                    urllib.request.urlretrieve(
                        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
                        model_path
                    )
                base_options = mp_python.BaseOptions(model_asset_path=model_path)
                options = mp_vision.FaceLandmarkerOptions(
                    base_options=base_options,
                    num_faces=1,
                )
                self.detector = mp_vision.FaceLandmarker.create_from_options(options)
                self._use_new = True
            except Exception:
                self._init_old_api()
        else:
            self._init_old_api()

    def _init_old_api(self):
        self._use_new = False
        mp_face = mp.solutions.face_mesh
        self.face_mesh = mp_face.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )

    def detect(self, img_bgr: np.ndarray) -> Optional[np.ndarray]:
        if self._use_new:
            return self._detect_new(img_bgr)
        return self._detect_old(img_bgr)

    def _detect_new(self, img_bgr: np.ndarray) -> Optional[np.ndarray]:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result = self.detector.detect(mp_image)
        if not result.face_landmarks:
            return None
        h, w = img_bgr.shape[:2]
        landmarks = []
        for lm in result.face_landmarks[0]:
            landmarks.append([int(lm.x * w), int(lm.y * h)])
        return np.array(landmarks, dtype=np.int32)

    def _detect_old(self, img_bgr: np.ndarray) -> Optional[np.ndarray]:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(img_rgb)
        if not results.multi_face_landmarks:
            return None
        face_landmarks = results.multi_face_landmarks[0]
        h, w = img_bgr.shape[:2]
        landmarks = []
        for landmark in face_landmarks.landmark:
            landmarks.append([int(landmark.x * w), int(landmark.y * h)])
        return np.array(landmarks, dtype=np.int32)

    def __del__(self):
        if hasattr(self, "face_mesh"):
            self.face_mesh.close()


def make_region_masks(landmarks: np.ndarray, img_shape: tuple) -> dict[str, np.ndarray]:
    """
    Create region masks for different facial areas.

    Args:
        landmarks: (N, 2) array of landmark coordinates
        img_shape: (height, width) of image

    Returns:
        Dict mapping region name to binary mask
    """
    h, w = img_shape[:2]
    masks = {}

    def create_mask_from_indices(indices, shape):
        mask = np.zeros(shape, dtype=np.uint8)
        if len(indices) == 0:
            return mask
        valid_indices = [i for i in indices if i < len(landmarks)]
        if len(valid_indices) < 3:
            return mask
        points = landmarks[valid_indices]
        hull = cv2.convexHull(points)
        cv2.fillConvexPoly(mask, hull, 255)
        return mask

    # Forehead
    forehead_indices = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                        172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

    # Nose
    nose_indices = [168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 164, 0, 11, 12,
                    13, 14, 15, 16, 17, 18, 200, 199, 175, 152]

    masks["forehead"] = create_mask_from_indices(forehead_indices[:20], (h, w))
    masks["nose"] = create_mask_from_indices(nose_indices, (h, w))
    masks["chin"] = create_mask_from_indices([152, 175, 200, 201, 18, 421, 406, 335, 273], (h, w))

    # Left cheek
    left_side = landmarks[landmarks[:, 0] < w // 2]
    if len(left_side) > 3:
        hull = cv2.convexHull(left_side)
        masks["left_cheek"] = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(masks["left_cheek"], hull, 255)

    # Right cheek
    right_side = landmarks[landmarks[:, 0] >= w // 2]
    if len(right_side) > 3:
        hull = cv2.convexHull(right_side)
        masks["right_cheek"] = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(masks["right_cheek"], hull, 255)

    # Combine cheeks
    if "left_cheek" in masks and "right_cheek" in masks:
        masks["cheeks"] = cv2.bitwise_or(masks["left_cheek"], masks["right_cheek"])
        del masks["left_cheek"]
        del masks["right_cheek"]

    return masks
