from typing import List

from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import squeezenet1_0
from torchvision import transforms

from telesto.model import ClassificationModelBase


def _convert_to_pil(array):
    if array.ndim == 3 and array.shape[-1] == 3:
        return Image.fromarray(array)

    if array.ndim == 3:
        one_channel = array[:, :, 0]
    else:  # only 2 and 3 dim arrays supported
        one_channel = array
    array = np.stack((one_channel,) * 3, axis=2)
    return Image.fromarray(array)


class ExampleClassificationModel(ClassificationModelBase):
    def __init__(self, model_path):
        super().__init__(classes=["on", "off"], model_path=model_path)
        self.input_size = 224
        self.data_transform = transforms.Compose(
            [
                transforms.Resize(self.input_size),
                transforms.CenterCrop(self.input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.model = self._load_model(model_path)

    def _load_model(self, model_path: str):
        model = squeezenet1_0()
        model.classifier[1] = nn.Conv2d(512, self.class_n, kernel_size=(1, 1), stride=(1, 1))
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
