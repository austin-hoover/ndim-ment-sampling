import os
import pathlib

import numpy as np
import scipy.ndimage
import matplotlib as mplt
import matplotlib.colors as colors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import seaborn as sns

from .utils import get_limits


def edges_to_coords_1d(edges: np.ndarray) -> np.ndarray:
    return 0.5 * (edges[:-1] + edges[1:])


def coords_to_edges_1d(coords: np.ndarray) -> np.ndarray:
    delta = np.diff(coords)[0]
    return np.hstack([coords - 0.5 * delta, [coords[-1] + 0.5 * delta]])


def edges_to_coords(edges: np.ndarray | list[np.ndarray]) -> np.ndarray | list[np.ndarray]:
    coords = None
    if np.isscalar(edges[0]):
        coords = edges_to_coords_1d(edges)
    else:
        coords = [edges_to_coords_1d(e) for e in edges]
    return coords


def coords_to_edges(coords: np.ndarray | list[np.ndarray]) -> np.ndarray | list[np.ndarray]:
    edges = None
    if np.isscalar(coords[0]):
        edges = coords_to_edges_1d(coords)
    else:
        edges = [coords_to_edges_1d(c) for c in coords]
    return edges


def get_colormap(name: str, left: float = 0.0, right: float = 1.0) -> colors.LinearSegmentedColormap:
    cmap = None
    if name in mplt.colormaps:
        cmap = mplt.colormaps[name]
    else:
        path = pathlib.Path(__file__)
        filename = path.parent.joinpath(f"cmaps/{name}.txt")
        if not os.path.exists(filename):
            raise FileNotFoundError
        cmap = colors.ListedColormap(np.loadtxt(filename))
    cmap = truncate_colormap(cmap, left=left, right=right)
    return cmap


def truncate_colormap(cmap, left=0.0, right=1.0, n=100):
    string = "trunc({n},{a:.2f},{b:.2f})".format(n=cmap.name, a=left, b=right)
    values = cmap(np.linspace(left, right, n))
    return colors.LinearSegmentedColormap.from_list(string, values)


def get_cubehelix_colormap(color: str = "red", dark: float = 0.20):
    kws = dict(
        n_colors=12,
        rot=0.0,
        gamma=1.0,
        hue=1.0,
        light=1.0,
        dark=dark,
        as_cmap=True,
    )

    cmap = None
    if color == "red":
        cmap = sns.cubehelix_palette(start=0.9, **kws)
    elif color == "pink":
        cmap = sns.cubehelix_palette(start=0.8, **kws)
    elif color == "blue":
        cmap = sns.cubehelix_palette(start=2.8, **kws)
    else:
        raise ValueError
    return cmap


def rms_ellipse_params(cov_matrix: np.ndarray) -> tuple[float, float, float]:
    sii = cov_matrix[0, 0]
    sjj = cov_matrix[1, 1]
    sij = cov_matrix[0, 1]

    angle = -0.5 * np.arctan2(2 * sij, sii - sjj)

    _sin = np.sin(angle)
    _cos = np.cos(angle)
    _sin2 = _sin**2
    _cos2 = _cos**2

    c1 = np.sqrt(abs(sii * _cos2 + sjj * _sin2 - 2 * sij * _sin * _cos))
    c2 = np.sqrt(abs(sii * _sin2 + sjj * _cos2 + 2 * sij * _sin * _cos))
    return (c1, c2, angle)


def plot_rms_ellipse(cov_matrix: np.ndarray, level: float = 1.0, ax=None, **kwargs) -> None:
    cx, cy, angle = rms_ellipse_params(cov_matrix)
    angle = -np.degrees(angle)
    center = (0.0, 0.0)
    cx *= 2.0 * level
    cy *= 2.0 * level
    patch = patches.Ellipse(center, cx, cy, angle=angle, fill=False, **kwargs)
    ax.add_patch(patch)


def plot_profile(
    values: np.ndarray,
    coords: list[np.ndarray] = None,
    edges: list[np.ndarray] = None,
    kind: str = "step",
    fill: bool = False,
    blur: float = 0.0,
    ax=None,
    **plot_kws
) -> None:
    if edges is not None:
        coords = edges_to_coords(edges)
    if coords is not None:
        edges = coords_to_edges(coords)

    values = np.copy(values)
    if blur:
        values = scipy.ndimage.gaussian_filter(values, sigma=blur)

    if kind == "step":
        return ax.stairs(values, edges, fill=fill, **plot_kws)
    elif kind == "line":
        if fill:
            return ax.fill_between(coords, np.zeros(len(values)), values, **plot_kws)
        else:
            return ax.plot(coords, values, **plot_kws)
    else:
        raise ValueError


def plot_hist_1d(
    x: np.ndarray,
    bins: int,
    limits: tuple[float, float],
    hist_kws: dict = None,
    ax=None,
    **plot_kws
) -> None:
    if hist_kws is None:
        hist_kws = {}

    values, edges = np.histogram(x, bins=bins, range=limits, **hist_kws)
    plot_profile(values=values, edges=edges, ax=ax, **plot_kws)


def plot_image(
    values: np.ndarray,
    coords: list[np.ndarray] = None,
    edges: list[np.ndarray] = None,
    kind: str = "pcolor",
    blur: float = 0.0,
    mask: bool = False,
    frac_thresh: float = None,
    log: bool = False,
    ax=None,
    **plot_kws
) -> None:

    plot_kws.setdefault("linewidth", 0.0)
    plot_kws.setdefault("rasterized", True)
    plot_kws.setdefault("shading", "auto")

    if edges is not None:
        coords = [0.5 * (edges[i][1:] + edges[i][:-1]) for i in range(len(edges))]

    if coords is None:
        coords = 2 * [np.arange(s) for s in values.shape]

    if blur:
        values = scipy.ndimage.gaussian_filter(values, sigma=blur)
    if frac_thresh:
        thresh = frac_thresh * np.max(values)
        values[values < thresh] = 0.0
    if mask:
        values = np.ma.masked_less_equal(values, 0.0)
    if log:
        values = np.log10(values + 1.00e-12)

    if kind == "pcolor":
        ax.pcolormesh(coords[0], coords[1], values.T, **plot_kws)
    elif kind == "contourf":
        ax.contourf(coords[0], coords[1], values.T, **plot_kws)
    elif kind == "contour":
        ax.contour(coords[0], coords[1], values.T, **plot_kws)
    else:
        raise ValueError


def plot_hist(
    x: np.ndarray,
    bins: int,
    limits: list[tuple[float, float]],
    rms_ellipse: bool = False,
    rms_ellipse_kws: dict = None,
    hist_kws: dict = None,
    ax=None,
    **plot_kws
) -> None:
    if hist_kws is None:
        hist_kws = {}

    values, edges = np.histogramdd(x, bins=bins, range=limits, **hist_kws)
    plot_image(values=values, edges=edges, ax=ax, **plot_kws)

    if rms_ellipse:
        if rms_ellipse_kws is None:
            rms_ellipse_kws = {}
        cov_matrix = np.cov(x.T)
        plot_rms_ellipse(cov_matrix, ax=ax, **rms_ellipse_kws)


def plot_corner(x: np.ndarray, fig_kws: dict = None, **plot_kws) -> tuple:
    if fig_kws is None:
        fig_kws = dict()

    grid = CornerGrid(ndim=x.shape[1], **fig_kws)
    grid.plot(x, **plot_kws)
    return grid.fig, grid.axs


class CornerGrid:
    def __init__(self, ndim: int, **fig_kws) -> None:
        self.ndim = ndim

        figwidth_default = 8.0 * (ndim / 6.0)
        fig_kws.setdefault("figsize", (figwidth_default, figwidth_default))

        fig, axs = plt.subplots(ncols=ndim, nrows=ndim, sharex=False, sharey=False, **fig_kws)
        for i in range(ndim):
            for ax in axs[i, 1:]:
                ax.set_yticks([])
            for ax in axs[:-1, i]:
                ax.set_xticks([])
            axs[i, i].set_ylim(0.0, 1.2)

        self.fig = fig
        self.axs = axs

    def set_limits(self, limits: list[tuple[float, float]]) -> None:
        for j in range(self.ndim):
            for i in range(self.ndim):
                ax = self.axs[i, j]
                ax.set_xlim(limits[j])
                if i != j:
                    ax.set_ylim(limits[i])

    def set_labels(self, labels: list[str], **kws) -> None:
        for i in range(self.ndim):
            self.axs[i, 0].set_ylabel(labels[i], **kws)
            self.axs[-1, i].set_xlabel(labels[i], **kws)
        self.fig.align_ylabels(self.axs[:, 0])
        self.fig.align_xlabels(self.axs[-1, :])

    def plot(
        self,
        x: np.ndarray,
        bins: int,
        limits: list[tuple[float, float]] = None,
        diag_kws: dict = None,
        **plot_kws
    ) -> None:
        if diag_kws is None:
            diag_kws = dict()
        diag_kws.setdefault("color", "black")
        diag_kws.setdefault("lw", 1.7)
        diag_kws.setdefault("blur", plot_kws.get("blur", 0.0))
        diag_kws.setdefault("color", plot_kws.get("colors", "black"))
        diag_kws.setdefault("lw", plot_kws.get("lineweights", 1.7))

        if limits is None:
            mins = np.min(x, axis=0)
            maxs = np.max(x, axis=0)
            limits = list(zip(mins, maxs))
            limits = np.array(limits)
            limits = limits * 1.33

        for i in range(self.ndim):
            for j in range(self.ndim):
                ax = self.axs[i, j]
                if i == j:
                    axis = i
                    values, edges = np.histogram(x[:, axis], bins=bins, range=limits[axis])
                    values = values / np.max(values)
                    plot_profile(values, edges=edges, ax=ax, **diag_kws)
                else:
                    axis = (j, i)
                    plot_hist(x[:, axis], bins=bins, limits=[limits[k] for k in axis], ax=ax, **plot_kws)
