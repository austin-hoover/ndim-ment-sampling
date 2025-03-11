"""Compare MENT to GPSR samples."""
import copy
import matplotlib.pyplot as plt
import ment
import numpy as np
import os
import pathlib
import sys
import torch
import zuko
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader
from typing import Callable

from data import load_data_flat
from flow import make_flow
from flow import train_flow
from flow import compute_flow_entropy
from flow import DataNormalizer
from model import ImageFactory
from model import LatticeFactory

sys.path.append("..")
from tools.plot import CornerGrid
from tools.plot import get_colormap
from tools.plot import get_cubehelix_colormap
from tools.plot import plot_image
from tools.utils import get_limits
from tools.utils import load_yaml_as_dict


plt.style.use("../tools/style.mplstyle")


# Setup
# --------------------------------------------------------------------------------------

path = pathlib.Path(__file__)
output_dir = os.path.join("outputs", path.stem)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# Helpers
# --------------------------------------------------------------------------------------

def make_data_loader(data: torch.Tensor) -> DataLoader:
    return DataLoader(data, batch_size=512, shuffle=True)


def make_data_normalizer(data: torch.Tensor) -> DataNormalizer:
    cov_matrix = torch.cov(data.T)
    return DataNormalizer(cov_matrix)


def estimate_entropy(flow: zuko.flows.Flow) -> torch.Tensor:
    return compute_flow_entropy(flow=flow, batch_size=100_000)


# Samples from reconstruction models
# --------------------------------------------------------------------------------------

models = {}
for key in ["gpsr", "ment"]:
    models[key] = {}
    for mode in ["train", "test"]:
        models[key][mode] = {"projections": [], "loss": None}


# GPSR
beam = torch.load("data/gpsr_paper_3d_scan_result.pt")
data = torch.vstack([beam.x, beam.px, beam.y, beam.py, beam.z, beam.pz]).T
models["gpsr"]["samp"] = torch.clone(data)


# MENT
model = ment.MENT(
    ndim=6,
    transforms=None,
    projections=None,
    sampler=None,
    prior=None,
)
model.load("outputs/train/checkpoint_05.pt")
model.sampler.shuffle = True
model.sampler.noise = 0.0
model.sampler.debug = False

cfg = load_yaml_as_dict("outputs/train/config.yaml")
scan_data = load_data_flat(cfg["data"])
scan_params = scan_data["train"]["params"]
lattice_factory = LatticeFactory(scan_params)

model.transforms = []
for index in range(len(scan_params)):
    model.transforms.append(lattice_factory.make_transform_np(index))

n = models["gpsr"]["samp"].shape[0]
data = model.unnormalize(model.sample(500_000))
data = data[-n:, :] 
data = torch.from_numpy(data)
data = data.float()
models["ment"]["samp"] = torch.clone(data)


# Load forward model and measured projections
# --------------------------------------------------------------------------------------

data = {}
for mode in ["train", "test"]:
    data[mode] = {"transforms": [], "images": [], "edges": []}

data_flat = load_data_flat("data/clipped_dset.pt", downscale=4)
for mode in ["train", "test"]:
    images = data_flat[mode]["images"]
    params = data_flat[mode]["params"]
    
    image_factory = ImageFactory(params, images)
    lattice_factory = LatticeFactory(params)
    
    projections, transforms, edges = [], [], []
    for index in range(len(images)):
        edges.append(image_factory.get_edges(index))
    
        projection = image_factory.get_values(index)
        projection = projection / torch.sum(projection)
        projections.append(projection)
    
        transform = lattice_factory.make_transform(index)
        transforms.append(transform)

    data[mode]["transforms"] = transforms
    data[mode]["projections"] = projections
    data[mode]["edges"] = edges


# Estimate loss
# --------------------------------------------------------------------------------------

# Simulate projections using each set of particles.
for key in ["ment", "gpsr"]:
    for mode in ["train", "test"]:        
        x = torch.clone(models[key]["samp"])
    
        projections_pred = []
        for i, transform in enumerate(data[mode]["transforms"]):
            x_out = transform(x)
            hist = torch.histogramdd(x_out[:, (0, 2)], data[mode]["edges"][i], density=True)
            projection = hist.hist
            projection = projection / torch.sum(projection)
            projections_pred.append(projection)
    
        models[key][mode]["projections"] = [torch.clone(projection) for projection in projections_pred]


# Compute average fractional error per pixel.
for mode in ["train", "test"]:
    for key in ["ment", "gpsr"]:
        for mode in ["train", "test"]:
            projections_meas = data[mode]["projections"]
            projections_pred = models[key][mode]["projections"]
        
            loss = 0.0
            for proj_pred, proj_meas in zip(projections_pred, projections_meas):
                loss += torch.mean(torch.abs(proj_meas - proj_pred)) / torch.max(proj_meas)
            loss /= len(projections_pred)

            models[key][mode]["loss"] = float(loss)

# Print results
for key in models:
    print(key)
    for mode in ["train", "test"]:
        print(f"loss ({mode}):", models[key][mode]["loss"])
    print()


# Estimate entropy
# --------------------------------------------------------------------------------------

limits = get_limits(models["gpsr"]["samp"].numpy(), rms=4.0, zero_center=True) 
limits = limits * 1000.0

for key in ["ment", "gpsr"]:
    print(key)

    # Get samples
    samples = torch.clone(models[key]["samp"])
    data_loader = make_data_loader(samples)
    data_normalizer = make_data_normalizer(samples)

    # Train flow and estimate entropy
    flow = make_flow()
    models[key]["flow_train_history"] = train_flow(flow, data_loader, data_normalizer)
    models[key]["flow_entropy"] = estimate_entropy(flow)

    # Plot training loss vs. iteration
    fig, ax = plt.subplots(figsize=(4, 2))
    ax.plot(models[key]["flow_train_history"]["loss"], color="black")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    fig.savefig(os.path.join(output_dir, f"fig_{key}_loss.pdf"))    

    # Plot true/flow samples
    with torch.no_grad():
        x_true = torch.clone(samples)
        x_true = x_true.numpy()
        
        x_pred = flow().sample((x_true.shape[0],))
        x_pred = data_normalizer.unnormalize(x_pred)
        x_pred = x_pred.numpy()
    
        scale = 1000.0
        x_pred = x_pred * scale
        x_true = x_true * scale
        
        bins = 50
        blur = 1.0
        
        grid = CornerGrid(ndim=6)
        grid.set_limits(limits)
        grid.plot(x_true, bins=bins, limits=limits, blur=blur)
        plt.savefig(os.path.join(output_dir, f"fig_{key}_corner_samp.pdf"))
        plt.close("all")
        
        grid = CornerGrid(ndim=6)
        grid.set_limits(limits)
        grid.plot(x_pred, bins=bins, limits=limits, blur=blur)
        plt.savefig(os.path.join(output_dir, f"fig_{key}_corner_flow.pdf"))
        plt.close("all")

# Print results
for key in models:
    print(key)
    print("entropy:", models[key]["flow_entropy"])
    for mode in ["train", "test"]:
        print(f"loss ({mode}):", models[key][mode]["loss"])
    print()


# Plot projections
# --------------------------------------------------------------------------------------

for mode in ["train", "test"]:
    print(mode)
    
    width = 1.5
    nplot = len(data[mode]["projections"])
    nrows = nplot // 2
    ncols = 6
    
    fig, axs = plt.subplots(
        ncols=ncols, 
        nrows=nrows,
        figsize=(width * ncols, width * nrows),
        sharex=True,
        sharey=True,
        gridspec_kw=dict(wspace=0.05, hspace=0.05)
    )
    for ax in axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        for loc in ax.spines:
            ax.spines[loc].set_visible(False)
    
    for i in range(nplot):
        values_meas = data[mode]["projections"][i].clone()
        values_pred_ment = models["ment"][mode]["projections"][i].clone()
        values_pred_gpsr = models["gpsr"][mode]["projections"][i].clone()
    
        # Make sure all normalized (should be already but double check)
        values_meas /= values_meas.sum()
        values_pred_ment /= values_pred_ment.sum()
        values_pred_gpsr /= values_pred_gpsr.sum()
        scale = values_meas.max()
    
        # Blur simulated images
        blur = 1.0
        values_pred_ment = gaussian_filter(values_pred_ment, blur)
        values_pred_gpsr = gaussian_filter(values_pred_gpsr, blur)
    
        # Plot
        for j, values in enumerate([values_meas, values_pred_ment, values_pred_gpsr]):
            row = i % nrows
            col = j + int(i >= nrows) * 3
            ax = axs[row, col]
            ax.pcolormesh(values.T, cmap=get_colormap("viridis"), shading="flat", rasterized=True, linewidth=0.0)
    
    fontsize = None
    axs[0, 0].set_title("DATA", fontsize=fontsize)
    axs[0, 1].set_title("MENT", fontsize=fontsize)
    axs[0, 2].set_title("GPSR", fontsize=fontsize)
    axs[0, 3].set_title("DATA", fontsize=fontsize)
    axs[0, 4].set_title("MENT", fontsize=fontsize)
    axs[0, 5].set_title("GPSR", fontsize=fontsize)
    plt.savefig(os.path.join(output_dir, f"fig_compare_data_{mode}.pdf"), dpi=350)
    plt.close("all")


# Plot distributions
# --------------------------------------------------------------------------------------

# [already done in entropy estimation step]
