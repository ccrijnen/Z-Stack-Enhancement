import h5py
import numpy as np
import zfocus as zf
from scipy.ndimage import gaussian_filter
import torchvision.transforms as T


def read_h5(path: str) -> np.ndarray:
    """
    Reads hdf5 file from a path string.

    Parameters
    ----------
    path : str
        Path to the hdf5 file.

    Returns
    -------
    image : np.ndarray
        NumPy ndarray containing a z_stack.
    """
    with h5py.File(path, "r") as file:
        z_stack = file["z_stack"][()]
    return z_stack


def get_transform(imsize=None, random_crop=False):
    t_list = [T.ToTensor()]
    if imsize is not None:
        if random_crop:
            t_list.append(T.RandomCrop(imsize))
        else:
            t_list.append(T.CenterCrop(imsize))
    return T.Compose(t_list)


def resample(vol):
    # Sliding window
    window_size = 128
    window_stride = 64
    win, x_range, y_range = zf.sliding_window_view(vol, (window_size,) * 2, stride=window_stride, axis=(0, 1),
                                                   return_ranges=True)

    thresh = 0.6
    bins = 24
    hist = zf.frequency_histogram(win, (3, 4), bins=bins)
    norm_hist = hist / hist.max(2, keepdims=True)  # normalize over z-axis

    sharpness = norm_hist[..., -8:].mean(-1)

    start_map, stop_map = z_range_from_threshold(sharpness, thresh, exact=True)

    # interpolate the sharpness back to original values
    shape = vol.shape[:2]
    full_start_map = zf.interpolate_index_map(start_map, shape, x_range, y_range)
    full_stop_map = zf.interpolate_index_map(stop_map, shape, x_range, y_range)

    # smooth the maps
    full_start_map = gaussian_filter(full_start_map, sigma=31)
    full_stop_map = gaussian_filter(full_stop_map, sigma=31)

    resampled_vol = zf.resample_vol_irregular(vol, full_start_map, full_stop_map, 20)
    return resampled_vol


def z_range_from_threshold(values, thresh, exact=False):
    """Z-range from percentile.

    Args:
        values: Array[d0, d1, ..., dn, bins].
        thresh: Threshold.
        exact: Whether to determine the exact indices as floats.

    Returns:
        Start indices, stop indices. Each as Array[d0, d1, ..., dn].
        Stop indices are one above largest included index.
    """
    values = np.array(values)
    max_z = values.shape[-1] - 1
    mask = values >= thresh
    res = start, stop = np.argmax(mask, axis=-1), max_z - np.argmax(mask[..., ::-1], axis=-1)
    if exact:
        res = ()
        for st in [[start, np.where(start-1 >= 0, start-1, 0)], [stop, np.where(stop+1 <= max_z, stop+1, max_z)]]:
            a = np.take_along_axis(values, st[0].reshape((values.shape[:-1]) + (1,)), axis=-1)
            b = np.take_along_axis(values, st[1].reshape((values.shape[:-1]) + (1,)), axis=-1)
            res += np.where((a - b) > 0, (thresh - b) / (a - b), 0).squeeze() + st[1],
    return res
