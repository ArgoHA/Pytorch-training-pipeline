from typing import Dict, Tuple

import cv2
import numpy as np
import pycuda.autoinit  # Initializes the CUDA driver
import pycuda.driver as cuda
import tensorrt as trt


class TensorRT_model:
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
        self.half = half  # Add half precision support
        self._load_engine()
        self.inputs, self.outputs, self.bindings, self.stream = self._allocate_buffers()
        self.context = self.engine.create_execution_context()
        self._init_params()

    def _init_params(self) -> None:
        if self.half:
            self.np_dtype = np.float16
        else:
            self.np_dtype = np.float32

    def _load_engine(self):
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(self.model_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

    def _allocate_buffers(self):
        inputs, outputs, bindings = [], [], []
        stream = cuda.Stream()

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            size = trt.volume(shape)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            # half-precision override
            if self.half and dtype == np.float32:
                dtype = np.float16

            # host + device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            bindings.append(int(device_mem))

            # classify input vs output
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_shape = shape
                inputs.append({"name": name, "host": host_mem, "device": device_mem})
            else:
                self.output_shape = shape
                outputs.append({"name": name, "host": host_mem, "device": device_mem})

        return inputs, outputs, bindings, stream

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        img = cv2.resize(
            image, (self.input_size[1], self.input_size[0]), interpolation=cv2.INTER_AREA
        )
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, then HWC to CHW
        img = np.ascontiguousarray(img, dtype=self.np_dtype)
        img = (img / 255.0).astype(self.np_dtype)
        img = (img - self.mean_norm[:, None, None]) / self.std_norm[:, None, None]
        return img[None]

    def _postprocess(self, outputs: np.ndarray) -> Tuple[str, float]:
        outputs = outputs.squeeze()  # Shape becomes (n_outputs,)
        # Convert outputs to float32 if they are in float16
        if outputs.dtype == np.float16:
            outputs = outputs.astype(np.float32)
        probs = np.exp(outputs) / np.sum(np.exp(outputs))
        label = int(np.argmax(probs))
        max_prob = float(probs[label])
        return label, max_prob

    def __call__(self, image: np.ndarray) -> Tuple[int, float]:
        input_data = self._preprocess(image).ravel()

        # copy to host
        np.copyto(self.inputs[0]["host"], input_data)

        # host→device
        cuda.memcpy_htod_async(self.inputs[0]["device"], self.inputs[0]["host"], self.stream)

        # bind addresses by tensor name
        for idx, addr in enumerate(self.bindings):
            name = self.engine.get_tensor_name(idx)
            self.context.set_tensor_address(name, addr)

        # run inference
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        # device→host
        cuda.memcpy_dtoh_async(self.outputs[0]["host"], self.outputs[0]["device"], self.stream)
        self.stream.synchronize()

        # reshape + postprocess
        out = self.outputs[0]["host"].reshape(self.output_shape)
        return self._postprocess(out)
