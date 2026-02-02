import os
import sys

import numpy as np


class PointQueue:
    def __init__(self, maxlen: int, init_value: np.ndarray):
        self.rec_queue: np.ndarray = np.full((maxlen, 3), np.nan)
        self.max_length: int = maxlen
        self.queue_tail: int = maxlen - 1

        self.enqueue(init_value)

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

    def __repr__(self):
        return "tail: " + str(self.queue_tail) + "\narray: " + str(self.rec_queue)

    def __str__(self):
        return "tail: " + str(self.queue_tail) + "\narray:\n" + str(self.rec_queue)


class PrintHider:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
