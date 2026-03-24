from abc import ABC, abstractmethod
import math

import cv2
import numpy as np
import torch
from torchvision.transforms import Pad


class ImagePreprocessor(ABC, torch.nn.Module):
    def __init__(self, imgsz: tuple[int, int] | list[int] = (480, 640)):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if isinstance(imgsz, list):
            if len(imgsz) == 2:
                self.imgsz = tuple(imgsz)
            else:
                raise ValueError('imgsz should be a list of length 2')
        elif isinstance(imgsz, tuple):
            self.imgsz = imgsz
        else:
            raise ValueError('imgsz should be a list or tuple of length 2')

    def pad_sequence(self, imgsz: tuple[int, int]):
        if imgsz[0] != imgsz[1]:
            top_pad = int(math.ceil((imgsz[1] - imgsz[0]) / 2.))
            bottom_pad = int(math.floor((imgsz[1] - imgsz[0]) / 2.))
            pad_sequence = [0, top_pad, 0, bottom_pad]
        else:
            pad_sequence = 0

        return pad_sequence

    @abstractmethod
    def preprocess(self, ir_frame):
        raise NotImplementedError

class GPUSquarePadImagePreprocessor(ImagePreprocessor):
    def __init__(self, imgsz: tuple[int, int] = (480, 640), pad: bool = True):
        super().__init__(imgsz)
        if self.device != 'cuda':
            raise RuntimeError('GPU required to use this preprocessor')

        self.pad_sequence = self.pad_sequence(imgsz=self.imgsz)
        self.top_pad = self.pad_sequence[1]

        if pad:
            self.pad = Pad(self.pad_sequence, fill=114)
        else:
            self.pad = torch.nn.Identity()

    def preprocess(self, ir_frame: np.ndarray):
        tensor = torch.from_numpy(ir_frame)
        tensor = tensor.to(self.device, non_blocking=True)
        tensor = self.pad(tensor)
        tensor = tensor.expand(1, 3, -1, -1).contiguous().float()

        tensor = tensor / 255.

        return tensor


class GPUImagePreprocessor(ImagePreprocessor):
    def __init__(self, imgsz: tuple[int, int] | list[int] = (480, 640), stride: int = 32, pad: bool = True):
        super().__init__(imgsz)
        if self.device != 'cuda':
            raise RuntimeError('GPU required to use this preprocessor')
        self.stride = stride

        self.pad_sequence = self.pad_sequence(imgsz=self.imgsz)
        self.top_pad = self.pad_sequence[1]

        if pad:
            self.pad = Pad(self.pad_sequence, fill=114)
        else:
            self.pad = torch.nn.Identity()

    def pad_sequence(self, imgsz: tuple[int, int]):
        input_h, input_w = imgsz

        target_h = int(math.ceil(input_h / self.stride) * self.stride)
        target_w = int(math.ceil(input_w / self.stride) * self.stride)

        pad_h = target_h - input_h
        pad_w = target_w - input_w

        top_pad = pad_h // 2
        bottom_pad = pad_h - top_pad

        left_pad = pad_w // 2
        right_pad = pad_w - left_pad

        return [left_pad, top_pad, right_pad, bottom_pad]


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
