from typing import List

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision

from telesto.models import ClassificationModelBase

CLASSES = ["healthy", "infected"]
MODEL_PATH = 'model.pt'


def _convert_to_pil(array):
    if array.ndim == 3 and array.shape[-1] == 3:
        return Image.fromarray(array)

    if array.ndim == 3:
        one_channel = array[:, :, 0]
    else:  # only 2 and 3 dim arrays supported
        one_channel = array
    array = np.stack((one_channel,) * 3, axis=2)
    return Image.fromarray(array)


class ClassificationModel(ClassificationModelBase):
    def __init__(self, model_path=MODEL_PATH):
        super().__init__(classes=CLASSES, model_path=model_path)
        self.input_size = 224
        self.data_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(self.input_size),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def _load_model(self, model_path: str):
        model = torchvision.models.resnet18()
        model.fc = nn.Linear(512, self.class_n)
        model.num_classes = self.class_n

        with open(model_path, "rb") as fp:
            model.load_state_dict(torch.load(fp))
        return model.eval()

    def predict(self, input_list: List[np.ndarray]) -> np.ndarray:
        with torch.no_grad():
            input_tensor = torch.stack(
                [self.data_transform(_convert_to_pil(array)) for array in input_list]
            )
            preds = nn.functional.softmax(self.model(input_tensor), dim=1).numpy()
            return preds
