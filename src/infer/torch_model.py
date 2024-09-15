from typing import Dict, Tuple

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image
from torch import nn
from torchvision import models


class Torch_model:
    def __init__(
        self,
        model_path: str,
        label_to_name: Dict[int, str],
        input_width: int = 256,
        input_height: int = 256,
        half: bool = False,
    ):
        self.mean_norm = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std_norm = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.input_size = (input_width, input_height)
        self.label_to_name = label_to_name
        self.n_outputs = len(label_to_name)
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.half = half
        self._init_params()

        self._load_model()
        self._test_pred()

    def _init_params(self) -> None:
        if self.half:
            self.np_dtype = np.float16
        else:
            self.np_dtype = np.float32

    def _load_model(self):
        self.model = models.shufflenet_v2_x1_5()
        self.model.fc = nn.Linear(in_features=1024, out_features=self.n_outputs)
        checkpoint = torch.load(self.model_path, weights_only=True)
        self.model.load_state_dict(checkpoint)

        if self.half:
            self.model.half()
        self.model.eval()
        self.model.to(self.device)

    def _test_pred(self):
        img = np.zeros((3, *self.input_size), dtype=self.np_dtype)
        self.model(torch.from_numpy(img).to(self.device)[None])

    def _preprocess(self, image: Image) -> torch.Tensor:
        image = image.resize(self.input_size)
        image = image.convert("RGB")
        image = np.array(image)
        image = (image / 255.0).astype(self.np_dtype)
        image = (image - self.mean_norm) / self.std_norm
        image = np.transpose(image, (2, 0, 1))  # HWC to CHW
        image = torch.from_numpy(image).to(self.device)[None]
        image = image.half() if self.half else image
        return image.to(self.device)

    def _postprocess(self, logits: torch.Tensor) -> Tuple[str, float]:
        probs = torch.softmax(logits, dim=1).cpu().detach().numpy()
        class_name = self.label_to_name[int(np.argmax(probs))]
        return class_name, np.max(probs)

    @torch.no_grad()
    def __call__(self, image: Image) -> Tuple[str, float, dict]:
        image = self._preprocess(image)
        logits = self.model(image)
        class_name, max_prob = self._postprocess(logits)
        return class_name, max_prob
