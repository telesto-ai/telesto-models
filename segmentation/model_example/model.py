from itertools import product

import numpy as np
import torch
import torchvision.transforms.functional as TF
from telesto.apps.segmentation import ImageStorage
from telesto.models import SegmentationModelBase
from unet import UNet

CLASSES = ["fg", "bg"]
MODEL_PATH = "model.pt"

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class SegmentationModel(SegmentationModelBase):
    def __init__(self, storage: ImageStorage):
        super().__init__(classes=CLASSES, model_path=MODEL_PATH, storage=storage)

    def _load_model(self, model_path: str) -> object:
        model = UNet(3, 3)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        return model.eval()

    def predict_tile(self, input: np.ndarray) -> np.ndarray:
        """
        Predicts an image tile.

        Args:
            input: numpy.ndarray of shape (height, width, num_channels)

        Returns:
            output: numpy.ndarray of shape (height, width)
        """
        input = TF.to_tensor(input).type(torch.FloatTensor).unsqueeze(0).to(device)
        prob_map = self.model(input)
        output = torch.argmax(prob_map, dim=1, keepdim=False)
        output = output.to("cpu").numpy()[0]
        return output

    def predict(self, input: np.ndarray, tile_res=(512, 512)) -> np.ndarray:
        """
        Predicts an entire image by splitting it up to tiles.

        Args:
             input: numpy.ndarray of shape (height, width, num_channels)
             tile_res: tuple containing the resolution of tiles

        Returns:
            output: numpy.ndarray of shape (height, width), containing the label-encoded
                mask
        """
        if input.ndim == 2:
            input = np.stack([input] * 3, axis=2)

        input_res = input.shape

        # placeholder for output
        output = np.zeros(shape=(input_res[0], input_res[1]))

        # generate tile coordinates
        tile_x = list(range(0, input_res[0], tile_res[0]))[:-1] + [input_res[0] - tile_res[0]]
        tile_y = list(range(0, input_res[1], tile_res[1]))[:-1] + [input_res[1] - tile_res[1]]
        tile = product(tile_x, tile_y)

        # predictions
        for slice in tile:
            in_tile = input[slice[0]:slice[0] + tile_res[0], slice[1]:slice[1] + tile_res[1], :]
            out_tile = self.predict_tile(in_tile)
            output[slice[0]:slice[0] + tile_res[0], slice[1]:slice[1] + tile_res[1]] = out_tile

        return output
