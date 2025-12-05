import os
from enum import Enum
from typing import Callable, Optional

import numpy as np
from torchvision.datasets import ImageFolder

from .extended import ExtendedVisionDataset


class Imagenette(ExtendedVisionDataset):
    class Split(Enum):
        TRAIN = "train"
        VAL = "val"

        @property
        def dirname(self) -> str:
            return self.value

    def __init__(
        self,
        *,
        split: "Imagenette.Split",
        root: str,
        extra: Optional[str] = None,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        dataset_root = os.path.join(root, split.dirname)
        if not os.path.isdir(dataset_root):
            raise RuntimeError(f'Imagenette split directory "{dataset_root}" not found')

        super().__init__(
            root=root,
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
        )

        self._split = split
        self._dataset_root = dataset_root
        self._dataset = ImageFolder(dataset_root)
        self._samples = self._dataset.samples
        self._targets = np.array(self._dataset.targets, dtype=np.int64)
        self._idx_to_class = {
            idx: class_id for class_id, idx in self._dataset.class_to_idx.items()
        }

    @property
    def split(self) -> "Imagenette.Split":
        return self._split

    def __len__(self) -> int:
        return len(self._samples)

    def _get_sample(self, index: int):
        try:
            return self._samples[index]
        except IndexError as e:
            raise RuntimeError(f"no sample {index}") from e

    def get_image_data(self, index: int) -> bytes:
        image_full_path, _ = self._get_sample(index)
        with open(image_full_path, mode="rb") as f:
            return f.read()

    def get_target(self, index: int) -> int:
        return int(self._targets[index])

    def get_targets(self) -> np.ndarray:
        return self._targets

    def get_class_id(self, index: int) -> str:
        target = self.get_target(index)
        return self._idx_to_class[target]

    def get_class_name(self, index: int) -> str:
        # Class folders already encode the semantic label (e.g. "n01440764").
        return self.get_class_id(index)

