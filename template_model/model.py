from typing import List

import numpy as np

from telesto.base_models import ClassificationModelBase


CLASSES = []            # add list of class labels here


class ClassificationModel(ClassificationModelBase):
    """
    Wrapper class for the model to be served.

    Attributes:
        classes: list, contains the labels
        classes_n: int, number of classes
        model_path: str, path to the model file
        model: the object representing the model, to be loaded with _load_model()
    """
    def __init__(self, model_path: str):
        """
        Add your initialization code here.
        Don't remove the super().__init__ call below!
        """
        super().__init__(classes=CLASSES, model_path=model_path)

    def _load_model(self, model_path: str):
        """
        Add the model loading script here. This method is called in the __init__
        and the return value is stored in the model attribute of the object.

        Args:
             model_path: str, the path to the model file.

        Returns:
            model: a callable object representing the machine learning model.
        """
        pass

    def predict(self, input_list: List[np.ndarray]) -> List[np.ndarray]:
        """
        Add the code for prediction here. The input is a list of numpy arrays, each
        of them is a single data point, for example a single image.

        The return value should be a list of numpy arrays, each element should contain
        the corresponding prediction probabilities.

        If your model produces labels but not prediciton probabilities, please one-hot
        encode the result. You can do this for instance with

        one_hot_encoded = np.eye(self.class_n)[pred]

        Args:
            input_list: List[numpy.ndarray], list of data points to be predicted

        Returns:
            pred: List[numpy.ndarray], list of numpy arrays containing prediction
                probabilties for each data point
        """
        pass
