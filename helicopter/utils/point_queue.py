"""
From here: https://stackoverflow.com/questions/42771110/fastest-way-to-left-cycle-a-numpy-array-like-pop-push-for-a-queue
"""
import numpy as np


class PointQueue:
    def __init__(self, maxlen: int):
        self.rec_queue: np.ndarray = np.full((maxlen, 3), np.nan)
        self.max_length: int = maxlen
        self.queue_tail: int = maxlen - 1

    def to_array(self) -> np.ndarray:
        head = (self.queue_tail + 1) % self.max_length
        return np.roll(self.rec_queue, -head, axis=0)  # this will force a copy

    def enqueue(self, new_data: np.ndarray) -> None:
        self.queue_tail = (self.queue_tail + 1) % self.max_length
        self.rec_queue[self.queue_tail] = new_data

    def peek(self):
        queue_head = (self.queue_tail + 1) % self.max_length
        return self.rec_queue[queue_head]

    def item_at(self, index: int):
        loc = (self.queue_tail + 1 + index) % self.max_length
        return self.rec_queue[loc]

    def replace_item_at(self, index: int, newItem: int):
        loc = (self.queue_tail + 1 + index) % self.max_length
        self.rec_queue[loc] = newItem

    def mean(self) -> np.ndarray:
        return np.nanmean(self.rec_queue, axis=0)

    def windowed_mean(self, window_type: str = 'hamming'):
        data = self.to_array()

        clean_data = data[~np.isnan(data).any(axis=1)]
        n = len(clean_data)

        if n < 2:
            return np.zeros(3)

        if window_type == 'hamming':
            window = np.hamming(n)
        elif window_type == 'hann':
            window = np.hanning(n)
        else:
            raise NotImplementedError

        windowed_data = clean_data * window[:, np.newaxis]
        windowed_mean = np.sum(windowed_data, axis=0) / np.sum(window)

        return windowed_mean

    def __repr__(self):
        return "tail: " + str(self.queue_tail) + "\narray: " + str(self.rec_queue)

    def __str__(self):
        return "tail: " + str(self.queue_tail) + "\narray:\n" + str(self.rec_queue)
