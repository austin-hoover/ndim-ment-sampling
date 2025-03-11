"""Train MENT model."""
import argparse
import os
import pathlib
import pickle
import sys
import time
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
import torch
from matplotlib import colors
from matplotlib import pyplot
from skimage.transform import downscale_local_mean

import ment
from ment.train import Evaluator
from ment.utils import unravel

# local
from data import load_data_flat
from model import ImageFactory
from model import LatticeFactory

sys.path.append("..")
from tools.plot import CornerGrid
from tools.plot import get_colormap
from tools.utils import save_dict_to_yaml

plt.style.use("../tools/style.mplstyle")


# Config
# --------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--debug", type=int, default=1)
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--tstamp", type=int, default=0)

parser.add_argument("--data", type=str, default="./data/clipped_dset.pt")
parser.add_argument("--im-start", type=int, default=0)
parser.add_argument("--im-stop", type=int, default=None)
parser.add_argument("--im-step", type=int, default=1)
parser.add_argument("--im-down", type=int, default=4)
parser.add_argument("--im-thresh", type=float, default=0.0)

parser.add_argument("--diag-blur", type=float, default=1.0)
parser.add_argument("--diag-thresh", type=float, default=0.01)
parser.add_argument("--diag-thresh-type", type=str, default="frac")

parser.add_argument("--cov", type=str, default="outputs/eval_gpsr/cov_matrix.dat")
parser.add_argument("--cov-block-diag", type=int, default=0)
parser.add_argument("--cov-diag", type=int, default=0)
parser.add_argument("--cov-update", type=int, default=0)

parser.add_argument("--prior-scale", type=float, default=1.1)

parser.add_argument("--samp", type=int, default=500_000)
parser.add_argument("--samp-burn", type=int, default=0)
parser.add_argument("--samp-chains", type=int, default=1000)
parser.add_argument("--samp-noise", type=float, default=0.05)
parser.add_argument("--samp-noise-type", type=str, default="gaussian")
parser.add_argument("--samp-start-scale", type=float, default=0.4)
parser.add_argument("--samp-start-find", type=int, default=0)
parser.add_argument("--samp-start-force-nonzero", type=int, default=0)
parser.add_argument("--samp-prop-dist", type=str, default="gaussian")
parser.add_argument("--samp-prop-scale", type=float, default=0.40)

parser.add_argument("--eval", type=int, default=1)
parser.add_argument("--eval-samp", type=int, default=None)

parser.add_argument("--plot", type=int, default=1)
parser.add_argument("--plot-samp", type=int, default=None)
parser.add_argument("--plot-freq", type=int, default=100)
parser.add_argument("--plot-bins", type=int, default=64)
parser.add_argument("--plot-blur", type=float, default=1.0)
parser.add_argument("--plot-ext", type=str, default="png")
parser.add_argument("--plot-dpi", type=float, default=300)

parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--epochs-coast", type=int, default=0)
parser.add_argument("--lr", type=float, default=0.25)
parser.add_argument("--stop-on-increase", type=int, default=1)
args = parser.parse_args()


# Setup
# --------------------------------------------------------------------------------------

path = pathlib.Path(__file__)
output_dir = os.path.join("outputs", path.stem)
if args.tstamp:
    timestamp = time.strftime("%y%m%d%H%M%S")
    output_dir = os.path.join(output_dir, timestamp)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

save_dict_to_yaml(vars(args), os.path.join(output_dir, "config.yaml"))

ndim = 6
rng = np.random.default_rng(args.seed)
cmap = get_colormap("mono", left=0.0, right=0.85)


# Load data and accelerator model
# --------------------------------------------------------------------------------------

data = load_data_flat(filename=args.data, downscale=args.im_down, thresh=args.im_thresh)
images = data["train"]["images"]
params = data["train"]["params"]

image_factory = ImageFactory(params, images)
lattice_factory = LatticeFactory(params)

# Collect projections
projections = []
for index in range(len(images)):
    coords = image_factory.get_coords_np(index)
    values = image_factory.get_values_np(index)
    projection = ment.HistogramND(
        values=values, 
        coords=coords, 
        axis=(0, 2), 
        blur=args.diag_blur, 
        thresh=args.diag_thresh,
        thresh_type=args.diag_thresh_type,
    )
    projection.normalize()
    projections.append([projection])

# Collect transforms
transforms = []
for index in range(len(images)):
    transform = lattice_factory.make_transform_np(index)
    transforms.append(transform)

# Select subset for testing
idx = slice(args.im_start, args.im_stop or len(images), args.im_step)
transforms = transforms[idx]
projections = projections[idx]


# Create MENT model
# --------------------------------------------------------------------------------------

# Load estimated covariance matrix and (optionally) zero off-diagonal elements.
cov_matrix = np.loadtxt(args.cov)

if args.cov_block_diag:
    cov_matrix_block_diag = [cov_matrix[i : i + 2, i : i + 2] for i in [0, 2, 4]]
    cov_matrix = scipy.linalg.block_diag(*cov_matrix_block_diag)

if args.cov_diag:
    cov_matrix_diag = np.diagonal(cov_matrix)
    cov_matrix = np.eye(ndim) * cov_matrix_diag

# Compute normalization matrix from estimated covariance matrix.
norm_matrix = ment.cov.normalization_matrix(cov_matrix, scale=True, block_diag=False)
unnorm_matrix = np.linalg.inv(norm_matrix)
cov_matrix_n = np.linalg.multi_dot([norm_matrix, cov_matrix, norm_matrix.T])

print("cov_matrix:")
print(np.round(cov_matrix * 1.00e06, 7))
print("cov_matrix_normalized:")
print(np.round(cov_matrix_n, 7))

# Prior (defined in normalized phase space)
prior = ment.GaussianPrior(ndim=ndim, scale=args.prior_scale)

# Sampler (defined in normalized phase space)
samp_burnin = args.samp_burn
samp_chains = args.samp_chains
samp_prop_cov = np.eye(ndim) * (args.samp_prop_scale**2)
samp_start = rng.normal(size=(samp_chains, ndim)) * args.samp_start_scale

sampler = ment.MetropolisHastingsSampler(
    ndim=ndim,
    chains=samp_chains,
    start=samp_start,
    proposal_cov=samp_prop_cov,
    burnin=samp_burnin,
    shuffle=True,
    verbose=True,
    debug=False,
    seed=args.seed,
    allow_zero_start=(not args.samp_start_force_nonzero),
    noise_scale=args.samp_noise,  # slight smoothing
    noise_type=args.samp_noise_type,
)

# MENT model
model = ment.MENT(
    ndim=ndim,
    transforms=transforms,
    projections=projections,
    prior=prior,
    unnorm_matrix=unnorm_matrix,
    sampler=sampler,
    interpolation_kws=dict(method="linear"),
    nsamp=args.samp,
    mode="sample",
    verbose=True,
)

if args.debug:
    print("Testing MCMC sampler with 10000 samples:")
    sampler.debug = True
    model.sample(10_000)


# Training
# --------------------------------------------------------------------------------------

def plot_proj(projections_pred: list[np.ndarray], projections_meas: list[np.ndarray]):
    n = len(projections_pred)
    ncols = min(10, n)
    nrows = int(np.ceil(n / ncols))
    nrows = nrows * 2

    fig, axs = plt.subplots(
        ncols=ncols,
        nrows=nrows,
        figsize=(ncols * 1.7, nrows * 1.7),
        gridspec_kw=dict(hspace=0, wspace=0),
    )

    index = 0
    for i in range(0, nrows, 2):
        for j in range(ncols):
            values_pred = projections_pred[index].values.copy()
            values_meas = projections_meas[index].values.copy()
            for ax, values in zip(axs[i: i + 2, j], [values_pred, values_meas]):
                values = values / np.max(values)
                ax.pcolormesh(values.T, cmap=cmap, linewidth=0.0, rasterized=True, shading="flat")
            index = index + 1

    for ax in axs.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        for loc in ax.spines:
            ax.spines[loc].set_visible(True)
    return fig, axs


def plot_dist(x: np.ndarray):
    scale = 1000.0

    rng = np.random.default_rng(args.seed)
    x_test = rng.normal(size=(100_000, ndim))
    x_test = model.unnormalize(x_test)
    xmax = np.std(x_test, axis=0) * 4.0

    xmax = xmax * scale

    limits = list(zip(-xmax, xmax))
    labels = [r"$x$", r"$p_x$", r"$y$", r"$p_y$", r"$z$", r"$p_z$"]

    grid = CornerGrid(ndim=ndim)
    grid.plot(x * scale, limits=limits, bins=args.plot_bins, blur=args.plot_blur, cmap=cmap)
    grid.set_labels(labels)
    return grid.fig, grid.axs


def plot_model(model):
    figs = []

    n = args.plot_samp or args.samp
    x = model.unnormalize(model.sample(n))

    projections_pred = unravel(ment.simulate_with_diag_update(x, model.transforms, model.diagnostics, blur=0.0))
    projections_meas = unravel(model.projections)

    fig, axs = plot_proj(projections_pred, projections_meas)
    figs.append(fig)

    fig, axs = plot_dist(x)
    figs.append(fig)

    return figs


eval_model = Evaluator(args.plot_samp or args.samp)
best_loss = np.inf
stop_training = False

for epoch in range(args.epochs + args.epochs_coast + 1):
    print(f"epoch = {epoch}")

    if (0 < epoch <= args.epochs):
        if args.samp_start_find:
            # Determining MCMC chain starting points
            model.sampler.start = np.zeros((args.samp_chains, ndim))
            model.sampler.burnin = 500
            model.sampler.start = model.sample(args.samp_chains)
            model.sampler.burnin = args.samp_burn 
    
        print("Updating Lagrange multiplier functions")
        model.gauss_seidel_step(learning_rate=args.lr)
    
    if args.eval:
        print("Evaluating")
        eval_result = eval_model(model)
        pprint(eval_result)
    
        filename = f"eval_{epoch:02.0f}.pkl"
        filename = os.path.join(output_dir, filename)
        with open(filename, "wb") as file:
            pickle.dump(eval_result, file)
    
        # Stop iterations if loss is increasing. (Technically it shouldn't but can 
        # because of noise.)
        loss = eval_result["discrepancy"]
        
        if loss < best_loss:
            best_loss = loss
        else:
            print("Loss increased! Stop training!")
            if args.stop_on_increase:
                print("Exiting program.")
                sys.exit()

    if args.plot:
        print("Plotting")
        for index, fig in enumerate(plot_model(model)):
            filename = f"fig_{index:02.0f}_{epoch:02.0f}.{args.plot_ext}"
            filename = os.path.join(output_dir, filename)
            fig.savefig(filename, dpi=args.plot_dpi)
        plt.close("all")

    # Update covariance matrix estimate and normalization matrix.
    if args.cov_update and args.eval:
        cov_matrix = eval_result["cov_matrix"]
        norm_matrix = ment.cov.normalization_matrix(cov_matrix, scale=True, block_diag=False)
        model.unnorm_matrix = np.linalg.inv(norm_matrix)

    # Save model
    state = {
        "transforms": None,  # can't pickle lattice
        "diagnostics": model.diagnostics,
        "projections": model.projections,
        "ndim": model.ndim,
        "prior": model.prior,
        "sampler": model.sampler,
        "unnorm_matrix": model.unnorm_matrix,
        "epoch": model.epoch,
        "lagrange_functions": model.lagrange_functions,
    }

    filename = f"checkpoint_{epoch:02.0f}.pt"
    filename = os.path.join(output_dir, filename)
    with open(filename, "wb") as file:
        pickle.dump(state, file, pickle.HIGHEST_PROTOCOL)
    print(filename)

