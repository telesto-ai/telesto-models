import numpy as np

from telesto.apps.segmentation import ImageStorage
from telesto.models import SegmentationModelBase

CLASSES = ["fg", "bg"]
MODEL_PATH = "model.pt"


class SegmentationModel(SegmentationModelBase):
    def __init__(self, storage: ImageStorage):
        super().__init__(classes=CLASSES, model_path=MODEL_PATH, storage=storage)

    def _load_model(self, model_path: str):
        pass

    def predict(self, input: np.ndarray) -> np.ndarray:
        output = input[:, :, 0] if input.ndim == 3 else input
        output /= output.max()
        output = output[output > 0.5].astype(np.uint8)
        return output
