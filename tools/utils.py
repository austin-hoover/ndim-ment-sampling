import os
import time

import numpy as np
import yaml


def list_paths(
    directory: str,
    startswith: str = None,
    notstartswith: str = ".",
    sort: bool = True,
    full_path: bool = True,
) -> list[str]:
    filenames = os.listdir(directory)
    if sort:
        filenames = sorted(filenames)
    if startswith:
        filenames = [f for f in filenames if f.startswith(startswith)]
    if notstartswith:
        if type(notstartswith) is not str:
            notstartswith = [notstartswith]
        for string in notstartswith:
            filenames = [f for f in filenames if not f.startswith(string)]
    if not full_path:
        return filenames
    return [os.path.join(directory, f) for f in filenames]


def save_dict_to_yaml(dictionary: dict, filename: str) -> None:
    with open(filename, "w") as file:
        yaml.dump(dictionary, file, default_flow_style=False)
        

def load_yaml_as_dict(filename: str) -> dict:
    with open(filename, "r") as file:
        return yaml.safe_load(file)


def get_limits(
    points: np.ndarray,
    rms: float = None,
    pad: float = 0.0,
    zero_center: bool = False,
    share: tuple[int, ...] | list[tuple[int, ...]] = None,
) -> np.ndarray:
    """Compute nice limits for binning/plotting.

    Parameters
    ----------
    points: np.ndarray, shape (..., n)
        Particle coordinates.
    rms : float
        If a number is provided, it is used to set the limits relative to the standard
        deviation of the distribution.
    pad : float
        Fractional padding to apply to the limits.
    zero_center : bool
        Whether to center the limits on zero.
    share : tuple[int] or list[tuple[int]]
        Limits are shared between the dimensions in each set. For example, if `share=(0, 1)`,
        axis 0 and 1 will share limits. Or if `share=[(0, 1), (4, 5)]` axis 0/1 will share
        limits, and axis 4/5 will share limits.

    Returns
    -------
    np.ndarray
        The limits [(xmin, xmax), (ymin, ymax), ...].
    """
    if points.ndim == 1:
        points = points[:, None]

    if rms is None:
        mins = np.min(points, axis=0)
        maxs = np.max(points, axis=0)
    else:
        means = np.mean(points, axis=0)
        stds = np.std(points, axis=0)
        widths = 2.0 * rms * stds
        mins = means - 0.5 * widths
        maxs = means + 0.5 * widths

    deltas = 0.5 * np.abs(maxs - mins)
    padding = deltas * pad
    mins = mins - padding
    maxs = maxs + padding
    limits = list(zip(mins, maxs))

    if share:
        if np.ndim(share[0]) == 0:
            share = [share]
        for axis in share:
            _min = min([limits[i][0] for i in axis])
            _max = max([limits[i][1] for i in axis])
            for i in axis:
                limits[i] = (_min, _max)

    if zero_center:
        mins, maxs = list(zip(*limits))
        maxs = np.max([np.abs(mins), np.abs(maxs)], axis=0)
        limits = list(zip(-maxs, maxs))

    if len(limits) == 1:
        limits = limits[0]

    limits = np.array(limits)

    return limits