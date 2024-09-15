from typing import Dict, Tuple

import numpy as np
import torch
from numpy.typing import NDArray
from openvino.runtime import Core
from PIL import Image


class OV_model:
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
        core = Core()
        det_ov_model = core.read_model(self.model_path)

        self.device_name = "CPU"
        if "GPU" in core.get_available_devices():
            self.device_name = "GPU"
        if self.device_name != "CPU":
            det_ov_model.reshape({0: [1, 3, *self.input_size]})

        self.model = core.compile_model(det_ov_model, self.device_name)

    def _test_pred(self) -> None:
        input_blob = np.zeros((1, 3, *self.input_size), dtype=self.np_dtype)
        self._predict(input_blob)

    def _predict(self, input_blob: NDArray) -> NDArray:
        result = self.model(input_blob)
        return result[self.model.output(0)]

    def _preprocess(self, image: Image) -> NDArray:
        image = image.resize(self.input_size)
        image = image.convert("RGB")
        image = np.array(image)
        image = (image / 255.0).astype(np.float32)
        image = (image - self.mean_norm) / self.std_norm
        image = np.transpose(image, (2, 0, 1))
        return image[None]

    def _postprocess(self, logits: torch.Tensor) -> Tuple[str, float]:
        probs = torch.softmax(logits, dim=1).cpu().detach().numpy()  # can do fully on np
        class_name = self.label_to_name[int(np.argmax(probs))]
        return class_name, np.max(probs)

    def __call__(self, image: Image) -> Tuple[str, float, dict]:
        image = self._preprocess(image)
        logits = self._predict(image)
        class_name, max_prob = self._postprocess(torch.from_numpy(logits))
        return class_name, max_prob
