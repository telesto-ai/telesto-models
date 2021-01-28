from typing import List

import numpy as np

from telesto.instance_segmentation import SegmentationObject
from telesto.instance_segmentation.model import SegmentationModelBase


class SegmentationModel(SegmentationModelBase):
    def __init__(self):
        super().__init__(classes=["fg", "bg"], model_path="")

    def _load_model(self, model_path: str):
        pass

    def predict(self, input: np.ndarray) -> List[SegmentationObject]:
        return [SegmentationObject(coords=[(0, 1), (0, 1)])]
