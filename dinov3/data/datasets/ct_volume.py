from pathlib import Path
from typing import Callable, List, Optional, Tuple, Any

import numpy as np
import torch
from torch.utils.data import Dataset


class CTVolumeDataset(Dataset):
    """
    Simple dataset for 3D volumetric CT data stored as NumPy binaries (.npy).

    Expected directory structure:
        root/
            *.npy          # each file is one volume, shape (D, H, W) or (C, D, H, W)

    Notes:
        - We currently ignore labels and return a dummy target (0) to fit the
          (image, target) protocol used elsewhere in the codebase.
        - Volumes are loaded fully into memory per __getitem__; for very large
          datasets you may want to add caching or memory‑mapping.
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        split: Optional[str] = None,
    ) -> None:
        """
        Args:
            root: Directory containing .npy volume files or subdirectories per split.
            transform: Transform applied to the image/volume only.
            target_transform: Transform applied to the target only.
            transforms: Transform applied to both (image, target).
            split: Optional split name (e.g. 'train', 'val'). If provided and
                a subdirectory with that name exists under root, we will look
                for .npy files there instead of directly under root.
        """
        super().__init__()

        root_path = Path(root)
        if split is not None:
            split_path = root_path / split
            if split_path.is_dir():
                root_path = split_path

        if not root_path.is_dir():
            raise ValueError(f"CTVolumeDataset root directory does not exist: {root_path}")

        self.root = str(root_path)
        self.transform = transform
        self.target_transform = target_transform
        self.transforms = transforms

        self._paths: List[str] = sorted(
            str(p) for p in Path(self.root).glob("*.npy") if p.is_file()
        )

        if len(self._paths) == 0:
            raise RuntimeError(f"No .npy volumes found in {self.root}")

    def __len__(self) -> int:
        return len(self._paths)

    def _load_volume(self, index: int) -> torch.Tensor:
        path = self._paths[index]
        arr = np.load(path)

        # Accept (D, H, W) or (C, D, H, W). Convert to torch.FloatTensor.
        if arr.ndim == 3:
            # assume single‑channel CT: (D, H, W) -> (1, D, H, W)
            arr = arr[None, ...]
        elif arr.ndim != 4:
            raise ValueError(
                f"Expected volume with 3 or 4 dimensions, got shape {arr.shape} for file {path}"
            )

        tensor = torch.from_numpy(arr).float()
        return tensor

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        volume = self._load_volume(index)

        # Dummy target to fit (image, target) interface.
        target: Any = 0

        # Apply torchvision‑style transforms if provided.
        if self.transforms is not None:
            volume, target = self.transforms(volume, target)
        else:
            if self.transform is not None:
                volume = self.transform(volume)
            if self.target_transform is not None:
                target = self.target_transform(target)

        return volume, target

