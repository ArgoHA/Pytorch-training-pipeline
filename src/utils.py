import random
from pathlib import Path
from typing import List

from torchvision import transforms


def get_class_names(data_path: Path) -> List[str]:
    class_names = [
        x.name
        for x in data_path.iterdir()
        if x.is_dir() and not str(x.name).startswith(".")
    ]
    return class_names


class RandomRotation90:
    def __init__(self, p: float = 0.05):
        self.p = p

    def __call__(self, x):
        if random.random() < self.p:
            return transforms.functional.rotate(x, 90)
        return x

    def __repr__(self):
        return self.__class__.__name__ + f"(p={self.p})"
