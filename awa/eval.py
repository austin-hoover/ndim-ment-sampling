"""Evaluate MENT model."""
import argparse
import os
import pathlib
import pickle
import sys
from pprint import pprint

import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import ment
import numpy as np
import scipy.ndimage
from omegaconf import OmegaConf

# local
from data import load_data_flat
from model import LatticeFactory

sys.path.append("..")
from tools.plot import CornerGrid
from tools.plot import get_colormap
from tools.plot import plot_hist
from tools.plot import plot_image
from tools.utils import list_paths
from tools.utils import load_yaml_as_dict
from tools.utils import get_limits

plt.style.use("../tools/style.mplstyle")
plt.rcParams["axes.linewidth"] = 1.20
plt.rcParams["text.usetex"] = True


# Arguments
# --------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", default=None)
parser.add_argument("--samp", type=int, default=2_000_000)

parser.add_argument("--bins", type=int, default=75)
parser.add_argument("--blur", type=float, default=1.0)
parser.add_argument("--diag-kind", type=str, default="line", choices=["step", "line"])
parser.add_argument("--diag-fill", type=int, default=1)
parser.add_argument("--limits-scale", type=float, default=4.0)

parser.add_argument("--dpi", type=float, default=300)
parser.add_argument("--debug", action="store_true")
parser.add_argument("--show", action="store_true")
args = parser.parse_args()


# Setup
# --------------------------------------------------------------------------------------

# Collect checkpoint files
input_dir = "outputs/train"
checkpoints = list_paths(input_dir, startswith="checkpoint")
pprint(checkpoints)

# Create output directory
path = pathlib.Path(__file__)
output_dir = os.path.join("outputs", path.stem)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load evaluation results
results = []
for filename in list_paths(input_dir, startswith="eval"):
    with open(filename, "rb") as file:
        result = pickle.load(file)
        results.append(result)

# Load config dict
cfg = load_yaml_as_dict(os.path.join(input_dir, "config.yaml"))
cfg = OmegaConf.create(cfg)


# Plot setup
# --------------------------------------------------------------------------------------

cmap = get_colormap("mono", left=0.0, right=0.85)
ndim = 6


# Plot loss vs. epoch
# --------------------------------------------------------------------------------------

epochs = np.arange(len(checkpoints))
losses = [result["discrepancy"] for result in results]
losses = losses[:len(epochs)]
losses = np.array(losses)

fig, ax = plt.subplots()
ax.plot(epochs, losses, marker=".")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")

if cfg.epochs_coast > 0:
    epoch_start=  cfg.epochs
    epoch_stop = epoch_start + cfg.epochs_coast
    ax.axvspan(epoch_start, epoch_stop, color="black", alpha=0.1)

filename = os.path.join(output_dir, "fig_loss.pdf")
plt.savefig(filename, dpi=args.dpi)
plt.close()


# Plot each checkpoint
# --------------------------------------------------------------------------------------

epochs = []
if args.epoch is None:
    epochs = [len(checkpoints) - 1]
elif args.epoch == "all":
    epochs = list(range(len(checkpoints)))
else:
    epochs = [args.epoch]


for epoch in reversed(epochs):
    print(f"PLOTTING EPOCH {epoch}")
    print(checkpoints[epoch])

    
    # Load model
    # ----------------------------------------------------------------------------------

    # Load parameters from pickled file.
    model = ment.MENT(
        ndim=6,
        transforms=None,
        projections=None,
        sampler=None,
        prior=None,
    )
    model.load(checkpoints[epoch])

    # Load transforms. (Can't pickle Bmad-X lattice: need to build.)
    data_flat = load_data_flat(cfg.data)
    params = data_flat["train"]["params"]
    factory = LatticeFactory(params)
    model.transforms = [factory.make_transform_np(i) for i in range(len(params))]

    # Simulate data
    x = model.unnormalize(model.sample(args.samp))
    projections_pred = ment.unravel(ment.simulate_with_diag_update(x, model.transforms, model.diagnostics, blur=0.0))
    projections_meas = ment.unravel(model.projections)

    
    # Plot distribution
    # ----------------------------------------------------------------------------------

    scale = 1000.0
    x_test = model.unnormalize(np.random.normal(size=(100_000, ndim)))
    x_test = x_test * scale
    limits = get_limits(x_test, rms=args.limits_scale)

    grid = CornerGrid(ndim=ndim, figsize=(7.0, 7.0))
    grid.plot(
        x * scale,
        limits=limits,
        bins=args.bins,
        blur=args.blur,
        cmap=cmap,
        diag_kws=dict(kind=args.diag_kind, fill=args.diag_fill, color="dimgray", alpha=1.0),
    )
    grid.set_labels([r"$x$", r"$p_x$", r"$y$", r"$p_y$", r"$z$", r"$p_z$"], fontsize="large")
    for ax in grid.axs.flat:
        for loc in ["top", "right"]:
            ax.spines[loc].set_visible(False)

    filename = f"fig_awa_corner_{epoch:02.0f}.pdf"
    filename = os.path.join(output_dir, filename)
    plt.savefig(filename)
    plt.close()
    

    # Plot projections
    # ----------------------------------------------------------------------------------

    nmeas = len(projections_pred)
    ncols = min(10, nmeas)
    nrows = int(np.ceil(nmeas / ncols))
    nrows = nrows * 2

    ax_size_x = 8.5 / ncols
    ax_size_y = ax_size_x

    fig, axs = plt.subplots(
        ncols=ncols,
        nrows=nrows,
        figsize=(ncols * ax_size_x, nrows * ax_size_y),
        gridspec_kw=dict(hspace=0, wspace=0),
    )

    index = 0
    for i in range(0, nrows, 2):
        for j in range(ncols):
            values_meas = projections_meas[index].values.copy()
            values_pred = projections_pred[index].values.copy()
            values_pred = scipy.ndimage.gaussian_filter(values_pred, args.blur)
            for ax, values in zip(axs[i: i + 2, j], [values_pred, values_meas]):
                values = values / np.max(values)    
                ax.pcolormesh(values.T, cmap=cmap, linewidth=0.0, rasterized=True, shading="flat")
            index = index + 1

    for ax in axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    for i in range(0, axs.shape[0], 2):
        for k, label in enumerate(["PRED", "MEAS"]):
            axs[i + k, 0].set_ylabel(label, fontsize="x-small")

    filename = f"fig_awa_proj_{epoch:02.0f}.pdf"
    filename = os.path.join(output_dir, filename)
    plt.savefig(filename)
    plt.close()
