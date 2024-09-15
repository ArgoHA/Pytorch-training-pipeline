import time
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf
from PIL import Image


class TFLiteModel:
    def __init__(
        self,
        model_path: str,
        label_to_name: Dict[int, str],
        input_width: int = 256,
        input_height: int = 256,
    ):
        self.num_threads = 2
        self.mean_norm = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std_norm = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.input_size = (input_width, input_height)
        self.label_to_name = label_to_name
        self.model_path = model_path

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

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        image = image.resize((self.input_shape[3], self.input_shape[2]))
        image = image.convert("RGB")
        image = np.array(image)
        image = (image / 255.0).astype(self.input_dtype)
        image = (image - self.mean_norm) / self.std_norm
        image = np.transpose(image, (2, 0, 1))  # HWC to CHW
        input_data = np.expand_dims(image, axis=0)
        return input_data

    def _postprocess(self, output_data: np.ndarray) -> Tuple[str, float]:
        if output_data.ndim == 2:
            output_data = output_data[0]

        probs = tf.nn.softmax(output_data).numpy()
        class_id = int(np.argmax(probs))
        class_name = self.label_to_name[class_id]
        confidence = float(np.max(probs))
        return class_name, confidence

    def __call__(self, image: Image.Image) -> Tuple[str, float]:
        input_data = self._preprocess(image)
        output_data = self._predict(input_data)
        class_name, confidence = self._postprocess(output_data)
        return class_name, confidence
