"""Face mesh detection using MediaPipe and region mask generation."""
import cv2
import numpy as np
from typing import Optional

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    NEW_API = True
except ImportError:
    NEW_API = False

if not NEW_API:
    import mediapipe as mp


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
            landmarks
