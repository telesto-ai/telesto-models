from typing import List

import numpy as np

from telesto.instance_segmentation import DataStorage, SegmentationObject
from telesto.instance_segmentation.model import SegmentationModelBase


class SegmentationModel(SegmentationModelBase):
    def __init__(self, storage: DataStorage):
        super().__init__(classes=["fg", "bg"], model_path="", storage=storage)

    def _load_model(self, model_path: str):
        pass

    def predict(self, input: np.ndarray) -> List[SegmentationObject]:
        return [SegmentationObject(class_i=1, x=10, y=10, w=2, h=2, mask=[[0, 1], [0, 1]])]
