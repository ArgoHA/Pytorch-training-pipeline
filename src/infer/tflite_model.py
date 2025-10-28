from typing import Dict, Tuple

import cv2
import numpy as np
import tensorflow as tf


class TFLiteModel:
    def __init__(
        self,
        model_path: str,
        n_outputs: int,
        input_size: Tuple[int, int] = (256, 256),  # (h, w)
    ):
        self.num_threads = 2
        self.mean_norm = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std_norm = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.input_size = input_size
        self.n_outputs = n_outputs
        self.model_path = model_path

        self.np_dtype = np.float32
        self._load_model()
        self._test_pred()

    def _load_model(self):
        # Load the TFLite model, allocate tensors and get input and output details
        self.interpreter = tf.lite.Interpreter(
            model_path=self.model_path, num_threads=self.num_threads
        )
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Extract input shape and data type.
        self.input_shape = self.input_details[0]["shape"]
        self.input_dtype = self.input_details[0]["dtype"]
        self.interpreter.allocate_tensors()

    def _test_pred(self):
        dummy_input = np.zeros(self.input_shape, dtype=self.input_dtype)
        self.interpreter.set_tensor(self.input_details[0]["index"], dummy_input)
        self.interpreter.invoke()

    def _predict(self, input_data: np.ndarray) -> np.ndarray:
        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_details[0]["index"])

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        img = cv2.resize(
            image, (self.input_size[0], self.input_size[1]), interpolation=cv2.INTER_AREA
        )  # (w, h)
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, then HWC to CHW
        img = np.ascontiguousarray(img, dtype=self.np_dtype)
        img = (img / 255.0).astype(self.np_dtype)
        img = (img - self.mean_norm[:, None, None]) / self.std_norm[:, None, None]
        return img[None]

    def _postprocess(self, output_data: np.ndarray) -> Tuple[str, float]:
        if output_data.ndim == 2:
            output_data = output_data[0]

        probs = tf.nn.softmax(output_data).numpy()
        label = int(np.argmax(probs))
        confidence = float(np.max(probs))
        return label, confidence

    def __call__(self, image: np.ndarray) -> Tuple[str, float]:
        input_data = self._preprocess(image)
        output_data = self._predict(input_data)
        label, confidence = self._postprocess(output_data)
        return label, confidence
