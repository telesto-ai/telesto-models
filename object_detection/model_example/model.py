from typing import List

import numpy as np

from telesto.utils import BBox
from telesto.object_detection import DetectionObject
from telesto.object_detection.model import ObjectDetectionModelBase


class ObjectDetectionModel(ObjectDetectionModelBase):
    def __init__(self):
        super().__init__(classes=[], model_path="")

    def _load_model(self, model_path: str):
        pass

    def predict(self, input: np.ndarray) -> List[DetectionObject]:
        return [DetectionObject(BBox(0, 0, 9, 9))]
