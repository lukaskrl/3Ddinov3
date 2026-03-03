# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import math
import random

import numpy as np
import torch

try:
    from scipy import ndimage
except Exception:
    ndimage = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


class MaskingGenerator:
    def __init__(
        self,
        input_size,
        num_masking_patches=None,
        min_num_patches=4,
        max_num_patches=None,
        min_aspect=0.3,
        max_aspect=None,
    ):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2
        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_masking_patches = num_masking_patches

        self.min_num_patches = min_num_patches
        self.max_num_patches = num_masking_patches if max_num_patches is None else max_num_patches

        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    def __repr__(self):
        repr_str = "Generator(%d, %d -> [%d ~ %d], max = %d, %.3f ~ %.3f)" % (
            self.height,
            self.width,
            self.min_num_patches,
            self.max_num_patches,
            self.num_masking_patches,
            self.log_aspect_ratio[0],
            self.log_aspect_ratio[1],
        )
        return repr_str

    def get_shape(self):
        return self.height, self.width

    def _mask(self, mask, max_mask_patches):
        delta = 0
        for _ in range(10):
            target_area = random.uniform(self.min_num_patches, max_mask_patches)
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < self.width and h < self.height:
                top = random.randint(0, self.height - h)
                left = random.randint(0, self.width - w)

                num_masked = mask[top : top + h, left : left + w].sum()
                # Overlap
                if 0 < h * w - num_masked <= max_mask_patches:
                    for i in range(top, top + h):
                        for j in range(left, left + w):
                            if mask[i, j] == 0:
                                mask[i, j] = 1
                                delta += 1

                if delta > 0:
                    break
        return delta

    def __call__(self, num_masking_patches=0):
        mask = np.zeros(shape=self.get_shape(), dtype=bool)
        mask_count = 0
        while mask_count < num_masking_patches:
            max_mask_patches = num_masking_patches - mask_count
            max_mask_patches = min(max_mask_patches, self.max_num_patches)

            delta = self._mask(mask, max_mask_patches)
            if delta == 0:
                break
            else:
                mask_count += delta

        return self.complete_mask_randomly(mask, num_masking_patches)

    def complete_mask_randomly(self, mask, num_masking_patches):
        shape = mask.shape
        m2 = mask.flatten()
        to_add = np.random.choice(np.where(~m2)[0], size=num_masking_patches - m2.sum(), replace=False)
        m2[to_add] = True
        return m2.reshape(shape)


def _otsu_threshold(volume: np.ndarray, bins: int = 256) -> float:
    v = volume.ravel()
    v = v[~np.isnan(v)]
    if v.size == 0:
        return 0.0
    hist, bin_edges = np.histogram(v, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    weight1 = np.cumsum(hist).astype(float)
    weight2 = np.cumsum(hist[::-1]).astype(float)[::-1]
    mean1 = np.cumsum(hist * bin_centers).astype(float)
    mean2 = (np.cumsum((hist * bin_centers)[::-1]).astype(float)[::-1])
    # avoid division by zero
    mean1 = np.where(weight1 > 0, mean1 / weight1, 0)
    mean2 = np.where(weight2 > 0, mean2 / weight2, 0)
    # between class variance
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    idx = np.nanargmax(variance12)
    return float(bin_centers[idx])


def crop_brain(
    volume,
    threshold: float = None,
    min_size: int = 1000,
):
    """Crop the largest foreground component (brain) from a 3D volume.

    Args:
        volume: 3D numpy array or torch tensor. If tensor has channel dim
                (`C, D, H, W`) the first channel is used.
        threshold: Optional intensity threshold. If None, Otsu is used.
        min_size: Minimum voxel count to accept a foreground component.

    Returns:
        cropped_volume: same type as input, cropped to the bounding box.
        brain_mask: boolean numpy array mask of the same spatial shape as input volume.
    """
    is_torch = isinstance(volume, torch.Tensor)
    if is_torch:
        vol_np = volume.detach().cpu().numpy()
    else:
        vol_np = np.asarray(volume)

    # Handle channel dimension if present
    if vol_np.ndim == 4:
        # assume (C, D, H, W) -> take first channel
        vol3 = vol_np[0]
    elif vol_np.ndim == 3:
        vol3 = vol_np
    else:
        raise ValueError("Expected 3D or 4D (C,D,H,W) volume")

    # Compute threshold
    if threshold is None:
        try:
            threshold = _otsu_threshold(vol3)
        except Exception:
            threshold = float(np.nanmean(vol3))

    mask = vol3 > threshold

    # Connected components: prefer scipy.ndimage if available
    if ndimage is not None:
        labels, num = ndimage.label(mask)
        if num == 0:
            # nothing found
            brain_mask = mask
        else:
            sizes = ndimage.sum(mask, labels, range(1, num + 1))
            if len(sizes) == 0:
                brain_mask = mask
            else:
                max_idx = int(np.argmax(sizes)) + 1
                if sizes[max_idx - 1] < min_size:
                    brain_mask = mask
                else:
                    brain_mask = labels == max_idx
    else:
        # Fallback: pick the largest bounding box from threshold mask
        coords = np.argwhere(mask)
        if coords.size == 0:
            brain_mask = mask
        else:
            # crude heuristic: connected components unavailable -> use full mask
            brain_mask = mask

    # Find bounding box
    idx = np.argwhere(brain_mask)
    if idx.size == 0:
        # return original
        if is_torch:
            return volume, brain_mask
        return vol_np, brain_mask

    z0, y0, x0 = idx.min(axis=0)
    z1, y1, x1 = idx.max(axis=0) + 1

    # Crop the original volume (preserve channels if present)
    if vol_np.ndim == 4:
        cropped = vol_np[:, z0:z1, y0:y1, x0:x1]
    else:
        cropped = vol_np[z0:z1, y0:y1, x0:x1]

    if is_torch:
        cropped = torch.as_tensor(cropped, device=volume.device, dtype=volume.dtype)

    return cropped, brain_mask


def crop_head(volume, background_pctile: float = 5.0, min_size: int = 1000, foreground_threshold: float = None):
    """Crop the whouele head (including skull) from background.

    Strategy (two modes):
        - If `foreground_threshold` is provided, consider voxels >= threshold
            as foreground candidates and select the largest connected component.
        - Otherwise, compute a background candidate mask by thresholding at the
            `background_pctile` intensity (low-intensity = air/background), find
            background components touching the volume border, invert to get head
            candidate, and take the largest connected component.

    Args:
            volume: 3D numpy array or torch tensor.
            background_pctile: Percentile used to find background intensity when
                                                    `foreground_threshold` is not provided.
            min_size: Minimum voxel count to accept a connected component as head.
            foreground_threshold: Optional absolute intensity threshold. Voxels
                                                        with value >= this are treated as foreground.

    Returns:
            cropped_volume, head_mask
    """

    is_torch = isinstance(volume, torch.Tensor)
    if is_torch:
        vol_np = volume.detach().cpu().numpy()
    else:
        vol_np = np.asarray(volume)

    if vol_np.ndim == 4:
        vol3 = vol_np[0]
    elif vol_np.ndim == 3:
        vol3 = vol_np
    else:
        raise ValueError("Expected 3D or 4D (C,D,H,W) volume")

    # Option A: explicit foreground threshold provided by caller
    if foreground_threshold is not None:
        fg_candidate = vol3 >= float(foreground_threshold)
        if ndimage is not None:
            labels, num = ndimage.label(fg_candidate)
            if num == 0:
                head_mask = fg_candidate
            else:
                sizes = ndimage.sum(fg_candidate, labels, range(1, num + 1))
                max_idx = int(np.argmax(sizes)) + 1
                head_mask = labels == max_idx
        else:
            # Fallback: use bounding box of fg_candidate
            coords = np.argwhere(fg_candidate)
            if coords.size == 0:
                head_mask = fg_candidate
            else:
                z0, y0, x0 = coords.min(axis=0)
                z1, y1, x1 = coords.max(axis=0) + 1
                head_mask = np.zeros_like(fg_candidate, dtype=bool)
                head_mask[z0:z1, y0:y1, x0:x1] = True

    else:
        # compute low-intensity threshold for background (air)
        v = vol3.ravel()
        v = v[~np.isnan(v)]
        if v.size == 0:
            raise ValueError("Empty volume")
        bg_thresh = float(np.percentile(v, background_pctile))

        bg_candidate = vol3 <= bg_thresh

        if ndimage is not None:
            labels, num = ndimage.label(bg_candidate)
            if num == 0:
                bg_mask = bg_candidate
            else:
                # find labels that appear on the border -> background
                border_labels = set()
                D, H, W = labels.shape
                # faces
                border_slices = [labels[0, :, :], labels[D - 1, :, :], labels[:, 0, :], labels[:, H - 1, :], labels[:, :, 0], labels[:, :, W - 1]]
                for sl in border_slices:
                    border_labels.update(np.unique(sl))
                border_labels.discard(0)
                if len(border_labels) == 0:
                    bg_mask = bg_candidate
                else:
                    mask_bg = np.zeros_like(labels, dtype=bool)
                    for lbl in border_labels:
                        mask_bg |= labels == lbl
                    bg_mask = mask_bg

            head_candidate = ~bg_mask
            # remove tiny isolated voxels in head_candidate by labeling and taking largest
            labels_head, num_h = ndimage.label(head_candidate)
            if num_h == 0:
                head_mask = head_candidate
            else:
                sizes = ndimage.sum(head_candidate, labels_head, range(1, num_h + 1))
                max_idx = int(np.argmax(sizes)) + 1
                head_mask = labels_head == max_idx
        else:
            # Fallback: use full non-background bounding box as coarse approximation
            coords = np.argwhere(bg_candidate == False)
            if coords.size == 0:
                head_mask = ~bg_candidate
            else:
                z0, y0, x0 = coords.min(axis=0)
                z1, y1, x1 = coords.max(axis=0) + 1
                head_mask = np.zeros_like(bg_candidate, dtype=bool)
                head_mask[z0:z1, y0:y1, x0:x1] = True

    idx = np.argwhere(head_mask)
    if idx.size == 0:
        if is_torch:
            return volume, head_mask
        return vol_np, head_mask

    z0, y0, x0 = idx.min(axis=0)
    z1, y1, x1 = idx.max(axis=0) + 1

    if vol_np.ndim == 4:
        cropped = vol_np[:, z0:z1, y0:y1, x0:x1]
    else:
        cropped = vol_np[z0:z1, y0:y1, x0:x1]

    if is_torch:
        cropped = torch.as_tensor(cropped, device=volume.device, dtype=volume.dtype)

    return cropped, head_mask


def visualize_mask(volume, mask, slice_index=None, axis: int = 0, cmap="gray", alpha: float = 0.4):
    """Visualize a single slice with mask overlay.

    Args:
        volume: 3D numpy array or torch tensor (or 4D with channel first).
        mask: 3D boolean numpy array.
        slice_index: index along `axis`; if None uses middle slice.
        axis: axis along which to slice (0=z,1=y,2=x).
    """
    if plt is None:
        raise RuntimeError("matplotlib is required for visualization")

    if isinstance(volume, torch.Tensor):
        vol = volume.detach().cpu().numpy()
    else:
        vol = np.asarray(volume)

    if vol.ndim == 4:
        vol = vol[0]

    if mask.ndim != 3:
        raise ValueError("mask must be 3D boolean array")

    D, H, W = mask.shape
    if slice_index is None:
        slice_index = (D // 2, H // 2, W // 2)[axis]

    if axis == 0:
        im = vol[slice_index]
        m = mask[slice_index]
    elif axis == 1:
        im = vol[:, slice_index, :]
        m = mask[:, slice_index, :]
    elif axis == 2:
        im = vol[:, :, slice_index]
        m = mask[:, :, slice_index]
    else:
        raise ValueError("axis must be 0,1,2")

    plt.figure(figsize=(6, 6))
    plt.imshow(im, cmap=cmap)
    plt.imshow(np.ma.masked_where(~m, m), cmap="Reds", alpha=alpha)
    plt.axis("off")
    plt.show()


class MaskingGenerator3D:
    """
    3D analogue of MaskingGenerator for volumetric patch grids.

    Operates on a (D, H, W) grid of patch indices and samples random cuboids
    until the requested number of masked patches is reached (up to
    `max_num_patches` per cuboid). The interface mirrors the 2D generator so
    it can be dropped into the existing collate logic.
    """

    def __init__(
        self,
        input_size,
        num_masking_patches=None,
        min_num_patches=4,
        max_num_patches=10,
    ):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 3
        assert len(input_size) == 3, "MaskingGenerator3D expects a (D, H, W) tuple for input_size"
        self.depth, self.height, self.width = input_size

        self.num_patches = self.depth * self.height * self.width
        self.num_masking_patches = num_masking_patches

        self.min_num_patches = min_num_patches
        self.max_num_patches = num_masking_patches if max_num_patches is None else max_num_patches

    def __repr__(self):
        repr_str = "Generator3D(%d, %d, %d -> [%d ~ %d], max = %d)" % (
            self.depth,
            self.height,
            self.width,
            self.min_num_patches,
            self.max_num_patches,
            self.num_masking_patches,
        )
        return repr_str

    def get_shape(self):
        return self.depth, self.height, self.width

    def _mask(self, mask, max_mask_patches):
        delta = 0
        for _ in range(10):
            target_volume = random.uniform(self.min_num_patches, max_mask_patches)

            # Sample approximate cuboid dimensions around the cubic root of the volume
            side = target_volume ** (1.0 / 3.0)
            d = int(round(side * random.uniform(0.5, 1.5)))
            h = int(round(side * random.uniform(0.5, 1.5)))
            w = int(round(side * random.uniform(0.5, 1.5)))

            if d < self.depth and h < self.height and w < self.width:
                zd = random.randint(0, self.depth - d)
                yh = random.randint(0, self.height - h)
                xw = random.randint(0, self.width - w)

                num_masked = mask[zd : zd + d, yh : yh + h, xw : xw + w].sum()
                volume = d * h * w
                # Overlap: accept if we add something and do not exceed max_mask_patches
                if 0 < volume - num_masked <= max_mask_patches:
                    for zz in range(zd, zd + d):
                        for yy in range(yh, yh + h):
                            for xx in range(xw, xw + w):
                                if mask[zz, yy, xx] == 0:
                                    mask[zz, yy, xx] = 1
                                    delta += 1

            if delta > 0:
                break
        return delta

    def __call__(self, num_masking_patches=0):
        mask = np.zeros(shape=self.get_shape(), dtype=bool)
        mask_count = 0
        while mask_count < num_masking_patches:
            max_mask_patches = num_masking_patches - mask_count
            max_mask_patches = min(max_mask_patches, self.max_num_patches)

            delta = self._mask(mask, max_mask_patches)
            if delta == 0:
                break
            else:
                mask_count += delta

        return self.complete_mask_randomly(mask, num_masking_patches)

    def complete_mask_randomly(self, mask, num_masking_patches):
        shape = mask.shape
        m2 = mask.flatten()
        to_add = np.random.choice(np.where(~m2)[0], size=num_masking_patches - m2.sum(), replace=False)
        m2[to_add] = True
        return m2.reshape(shape)

class MaskingGenerator3d:

    def __init__(
        self,
        input_size
    ):
        """
        Create a masking generator for 3D data, uses uniform random sampling to mask patches.
        Args:
            input_size: Size of the input data.
        """
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 3
        self.height, self.width, self.depth = input_size
        self.num_patches = self.height * self.width * self.depth

    def __repr__(self):
        repr_str = "Generator(%d, %d, %d)" % (
            self.height,
            self.width,
            self.depth
        )
        return repr_str

    def get_shape(self):
        return self.height, self.width, self.depth

    def _mask(self, mask, n_masked):

        mask_inds = random.sample(range(self.num_patches), k=n_masked)
        mask.ravel()[mask_inds] = 1

    def __call__(self, num_masking_patches=0):
        mask = np.zeros(shape=self.get_shape(), dtype=bool)
        self._mask(mask, num_masking_patches)
        return mask