from typing import Dict, Tuple

import numpy as np
import pycuda.autoinit  # Initializes the CUDA driver
import pycuda.driver as cuda
import tensorrt as trt
from PIL import Image


class TensorRT_model:
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
        self.half = half  # Add half precision support
        self._load_engine()
        self._allocate_buffers()
        self.context = self.engine.create_execution_context()

    def _load_engine(self):
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(self.model_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

    def _allocate_buffers(self):
        self.inputs = []
        self.outputs = []
        self.bindings = [None] * self.engine.num_bindings
        self.stream = cuda.Stream()
        for binding in self.engine:
            binding_idx = self.engine.get_binding_index(binding)
            shape = self.engine.get_binding_shape(binding)
            size = trt.volume(shape)
            # Get the data type from the engine binding
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            # Adjust dtype for half precision if applicable
            if self.half and dtype == np.float32:
                dtype = np.float16
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings at the correct index
            self.bindings[binding_idx] = int(device_mem)
            if self.engine.binding_is_input(binding):
                self.input_shape = shape
                self.inputs.append(
                    {
                        "host": host_mem,
                        "device": device_mem,
                        "binding": binding_idx,
                        "shape": shape,
                    }
                )
            else:
                self.output_shape = shape
                self.outputs.append(
                    {
                        "host": host_mem,
                        "device": device_mem,
                        "binding": binding_idx,
                        "shape": shape,
                    }
                )

    def _preprocess(self, image: Image) -> np.ndarray:
        image = image.resize(self.input_size)
        image = image.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = (image - self.mean_norm) / self.std_norm
        image = np.transpose(image, (2, 0, 1))  # HWC to CHW
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        if self.half:
            image = image.astype(np.float16)
        return image

    def _postprocess(self, outputs: np.ndarray) -> Tuple[str, float]:
        outputs = outputs.squeeze()  # Shape becomes (n_outputs,)
        # Convert outputs to float32 if they are in float16
        if outputs.dtype == np.float16:
            outputs = outputs.astype(np.float32)
        probs = np.exp(outputs) / np.sum(np.exp(outputs))
        class_idx = int(np.argmax(probs))
        class_name = self.label_to_name[class_idx]
        max_prob = float(probs[class_idx])
        return class_name, max_prob

    def __call__(self, image: Image) -> Tuple[str, float]:
        # Preprocess the image
        input_data = self._preprocess(image)
        # Copy input data to host input buffer
        np.copyto(self.inputs[0]["host"], input_data.ravel())
        # Transfer input data to the GPU.
        cuda.memcpy_htod_async(self.inputs[0]["device"], self.inputs[0]["host"], self.stream)
        # Run inference.
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(self.outputs[0]["host"], self.outputs[0]["device"], self.stream)
        # Synchronize the stream
        self.stream.synchronize()
        # Postprocess results
        output_data = self.outputs[0]["host"]
        output_shape = self.outputs[0]["shape"]
        output_data = output_data.reshape(output_shape)
        class_name, max_prob = self._postprocess(output_data)
        return class_name, max_prob
