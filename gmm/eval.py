import argparse
import os
import pathlib
import pickle
import sys

import matplotlib.pyplot as plt
import ment
import numpy as np
import scipy.ndimage
from omegaconf import DictConfig
from omegaconf import OmegaConf

sys.path.append("..")
from tools.plot import get_cubehelix_colormap
from tools.plot import get_colormap
from tools.plot import CornerGrid
from tools.plot import plot_image
from tools.utils import list_paths
from tools.utils import load_yaml_as_dict


plt.style.use("../tools/style.mplstyle")
plt.rcParams["text.usetex"] = True


# Setup
# --------------------------------------------------------------------------------------

input_dir = "./outputs/train"

output_dir = "./outputs/eval"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

cfg = load_yaml_as_dict(os.path.join(input_dir, "config.yaml"))
cfg = OmegaConf.create(cfg)


# Define source distribution
# --------------------------------------------------------------------------------------

dist = ment.dist.get_dist(cfg.dist, ndim=cfg.ndim, seed=cfg.seed)
x_true = dist.sample(cfg.dist_size)


# Load model
# --------------------------------------------------------------------------------------

checkpoint_paths = list_paths(os.path.join(input_dir, "checkpoints"))
checkpoint_path = checkpoint_paths[-1]

model = ment.MENT(
    ndim=cfg.ndim, 
    transforms=None,
    projections=None,
    prior=None,
    sampler=None,
)
model.load(checkpoint_path)


# Simulate data
# --------------------------------------------------------------------------------------

x_pred = model.unnormalize(model.sample(5_000_000))
projections_true = ment.unravel(model.projections)
projections_pred = ment.unravel(
    ment.simulate_with_diag_update(x_pred, model.transforms, model.diagnostics, blur=0.0)
)


# Plot
# --------------------------------------------------------------------------------------

cmap = get_colormap("mono", right=0.85)
width = 0.75

fig, axs = plt.subplots(
    ncols=15, 
    nrows=2,
    figsize=(width * 15.0, width * 2.0),
    sharex=True,
    sharey=True,
    gridspec_kw=dict(wspace=0, hspace=0)
)
for ax in axs.flat:
    ax.set_xticks([])
    ax.set_yticks([])

for j in range(axs.shape[1]):
    proj_true = projections_true[j].copy()
    proj_pred = projections_pred[j].copy()
    scale = proj_true.values.max()

    proj_pred.values = scipy.ndimage.gaussian_filter(proj_pred.values, 1.0)

    for i, proj in enumerate([proj_true, proj_pred]):
        ax = axs[i, j]
        plot_image(
            values=proj.values.T / scale,
            coords=proj.coords,
            cmap=cmap,
            ax=ax,
        )

for ax, label in zip(axs[:, 0], ["PRED", "TRUE"]):
    ax.set_ylabel(label)

# dims = [r"$x$", r"$p_z$", r"$y$", r"$p_y$", r"$z$", r"$p_z$"]
dims = [r"$x_1$", r"$x_2$", r"$x_3$", r"$x_4$", r"$x_5$", r"$x_6$"]
index = 0
for i in range(6):
    for j in range(i + 1, 6):
        title = f"{dims[i]}-{dims[j]}"
        axs[0, index].set_title(title, fontsize="large")
        index += 1
    
plt.savefig(os.path.join(output_dir, "fig_gmm_flat.pdf"))


