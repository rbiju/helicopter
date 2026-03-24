import torch
import numpy as np

from ultralytics import YOLO

from .preprocessor import ImagePreprocessor, GPUImagePreprocessor


class HelicopterYOLO(torch.nn.Module):
    def __init__(self, model: YOLO, preprocessor: ImagePreprocessor = GPUImagePreprocessor(),
                 imgsz: tuple[int, int] | list[int] = (480, 640), conf: float = 0.25):
        super().__init__()
        self.model = model
        self.preprocessor = preprocessor
        self.conf = conf

        if isinstance(imgsz, list):
            if len(imgsz) == 2:
                self.imgsz = tuple(imgsz)
            else:
                raise ValueError('imgsz should be a list of length 2')
        elif isinstance(imgsz, tuple):
            self.imgsz = imgsz
        else:
            raise ValueError('imgsz should be a list or tuple of length 2')

        dummy_in = np.random.rand(*self.imgsz)
        dummy_tensor = self.preprocessor.preprocess(dummy_in)
        _ = self.model.predict(dummy_tensor, verbose=False, device=0, half=True)

    def forward(self, ir_image: np.ndarray):
        tensor = self.preprocessor.preprocess(ir_image)
        results = self.model.predict(tensor, verbose=False, device=0, conf=self.conf, half=True)

        boxes = results[0].boxes.data.clone()
        boxes[:, [1, 3]] -= self.preprocessor.top_pad
        boxes = boxes[:, :4].cpu().numpy().astype(int)
        return boxes
