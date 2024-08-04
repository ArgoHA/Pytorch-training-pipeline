from typing import Tuple

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models


class CustomModel:
    MEAN_NORM = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD_NORM = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    INPUT_SIZE = [256, 256]
    N_OUTPUTS = 3
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CLASSES = {0: "class_1", 1: "class_2", 2: "class_3"}

    def __init__(self, model_path: str):
        self.model = models.efficientnet_b0()
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features=1280, out_features=self.N_OUTPUTS),
        )
        checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        self.model.to(self.DEVICE)

        self._test_pred()

    def _test_pred(self):
        img = torch.rand(1, self.N_OUTPUTS, *self.INPUT_SIZE, device=self.DEVICE)
        self.model(img)

    def _preprocess(self, image: Image) -> torch.Tensor:
        image = image.resize(self.INPUT_SIZE)
        image = image.convert("RGB")
        image = np.array(image)
        image = (image / 255.0).astype(np.float32)
        image = (image - self.MEAN_NORM) / self.STD_NORM
        image = np.transpose(image, (2, 0, 1))
        image = torch.from_numpy(image).to(self.DEVICE)[None]
        return image.to(self.DEVICE)

    def _postprocess(self, logits: torch.Tensor) -> Tuple[str, float, dict]:
        probs = torch.softmax(logits, dim=1).cpu().detach().numpy()
        class_name = self.CLASSES[np.argmax(probs)]
        probs_dict = {self.CLASSES[i]: prob for i, prob in enumerate(probs.flatten())}
        return class_name, np.max(probs), probs_dict

    @torch.no_grad()
    def process(self, image: Image) -> Tuple[str, float, dict]:
        """
        Process images and detect image type

        Args:
            image (Image): PIL image

        Returns:
            Tuple[str, float, dict]: (image class, score, probs dicts)
        """
        image = self._preprocess(image)
        logits = self.model(image)
        class_name, max_prob, probs_dict = self._postprocess(logits)
        return class_name, max_prob, probs_dict
