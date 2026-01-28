from abc import ABC, abstractmethod
import math

import torch
from torchvision.transforms import Pad
import numpy as np
import cv2


class ImagePreprocessor(ABC, torch.nn.Module):
    def __init__(self, imgsz: tuple[int, int] = (480, 640)):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if imgsz[0] != imgsz[1]:
            top_pad = int(math.ceil((imgsz[1] - imgsz[0]) / 2.))
            bottom_pad = int(math.floor((imgsz[1] - imgsz[0]) / 2.))
            pad_sequence = [0, top_pad, 0, bottom_pad]
        else:
            pad_sequence = 0
            top_pad = 0

        self.pad_sequence = pad_sequence
        self.top_pad = top_pad

    @abstractmethod
    def preprocess(self, ir_frame):
        raise NotImplementedError


class GPUImagePreprocessor(ImagePreprocessor):
    def __init__(self, imgsz: tuple[int, int] = (480, 640)):
        super().__init__(imgsz)
        if self.device != 'cuda':
            raise RuntimeError('GPU required to use this preprocessor')

        self.pad = Pad(self.pad_sequence, fill=114)

    def preprocess(self, ir_frame: np.ndarray):
        tensor = torch.from_numpy(ir_frame)
        tensor = tensor.to(self.device, non_blocking=True)
        tensor = self.pad(tensor)

        tensor = tensor.expand(1, 3, -1, -1).contiguous().float()

        tensor = tensor / 255.
        return tensor


class CPUImagePreprocessor(ImagePreprocessor):
    def __init__(self, imgsz: tuple[int, int] = (480, 640)):
        super().__init__(imgsz)

    def preprocess(self, ir_frame: np.ndarray):
        out = cv2.cvtColor(ir_frame, cv2.COLOR_GRAY2BGR)
        return out
