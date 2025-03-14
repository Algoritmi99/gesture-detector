from collections import deque

import numpy as np


class Buffer(object):
    def __init__(self, buffer_size: int):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)

    def add(self, obj: np.ndarray):
        self.buffer.append(obj)

    def get_all(self):
        return np.array(list(self.buffer))

    def get(self):
        return self.get_all() if len(self.buffer) == self.buffer_size else None

    def get_flatten(self):
        out = self.get()
        return out.flatten() if out is not None else None

    def __str__(self):
        return str(self.get_all())
