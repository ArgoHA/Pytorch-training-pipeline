from typing import Dict, Tuple

import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from openvino.runtime import Core


class OV_model:
    def __init__(
        self,
        model_path: str,
        n_outputs: int,
        input_size: Tuple[int, int] = (256, 256),  # (h, w)
        half: bool = False,
    ):
        self.mean_norm = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std_norm = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.input_size = input_size
        self.n_outputs = n_outputs
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

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        img = cv2.resize(
            image, (self.input_size[1], self.input_size[0]), interpolation=cv2.INTER_AREA
        )
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, then HWC to CHW
        img = np.ascontiguousarray(img, dtype=self.np_dtype)
        img = (img / 255.0).astype(self.np_dtype)
        img = (img - self.mean_norm[:, None, None]) / self.std_norm[:, None, None]
        return img[None]

    def _postprocess(self, logits: torch.Tensor) -> Tuple[str, float]:
        probs = torch.softmax(logits, dim=1).cpu().detach().numpy()  # can do fully on np
        label = int(np.argmax(probs))
        return label, np.max(probs)

    def __call__(self, image: np.ndarray) -> Tuple[str, float, dict]:
        image = self._preprocess(image)
        logits = self._predict(image)
        label, max_prob = self._postprocess(torch.from_numpy(logits))
        return label, max_prob
