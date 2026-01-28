import torch
import numpy as np

from ultralytics import YOLO

from .preprocess import ImagePreprocessor, GPUImagePreprocessor


class HelicopterYOLO(torch.nn.Module):
    def __init__(self, model: YOLO, preprocessor: ImagePreprocessor = GPUImagePreprocessor(), imgsz: tuple[int, int] = (480, 640), conf: float = 0.25):
        super().__init__()
        self.model = model
        self.preprocessor = preprocessor
        self.conf = conf

        dummy_in = np.random.rand(*imgsz)
        dummy_tensor = self.preprocessor.preprocess(dummy_in)
        _ = self.model.predict(dummy_tensor, verbose=False, device=0, half=True)

    def forward(self, ir_image: np.ndarray):
        tensor = self.preprocessor.preprocess(ir_image)
        results = self.model.predict(tensor, verbose=False, device=0, conf=self.conf, half=True)

        return results
