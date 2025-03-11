import argparse
import os
import pathlib
import pickle
import shutil
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import ment
import ment.train

sys.path.append("..")
from tools.utils import list_paths
from tools.utils import save_dict_to_yaml


# Config
# --------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--ndim", type=int, default=6)
parser.add_argument("--dist", type=str, default="gaussian-mixture")
parser.add_argument("--dist-size", type=int, default=1_000_000)
parser.add_argument("--diag-xmax", type=float, default=3.5)
parser.add_argument("--diag-bins", type=int, default=64)
parser.add_argument("--diag-blur", type=float, default=0.0)
parser.add_argument("--diag-thresh", type=float, default=0.005)
parser.add_argument("--diag-thresh-type", type=str, default="frac")
parser.add_argument("--samp-burn", type=int, default=500)
parser.add_argument("--samp-chains", type=int, default=1000)
parser.add_argument("--samp-start-scale", type=float, default=0.5)
parser.add_argument("--samp-prop-dist", type=str, default="gaussian")
parser.add_argument("--samp-prop-scale", type=float, default=0.5)
parser.add_argument("--samp-noise", type=float, default=0.05)
parser.add_argument("--plot-samp", type=int, default=1_000_000)
parser.add_argument("--plot-bins", type=int, default=64)
parser.add_argument("--plot-blur", type=float, default=0.0)
parser.add_argument("--prior-scale", type=float, default=1.0)
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--lr", type=float, default=0.90)
parser.add_argument("--samp", type=int, default=1_000_000)
args = parser.parse_args()


# Setup
# --------------------------------------------------------------------------------------
    
path = pathlib.Path(__file__)
output_dir = os.path.join("outputs", path.stem)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

save_dict_to_yaml(vars(args), os.path.join(output_dir, "config.yaml"))

ndim = args.ndim
rng = np.random.default_rng(args.seed)


# Initial distribution
# --------------------------------------------------------------------------------------  

dist = ment.dist.get_dist(args.dist, ndim=args.ndim, seed=args.seed)
x_true = dist.sample(args.dist_size)


# Forward model
# --------------------------------------------------------------------------------------

# Settings
nmeas = ndim * (ndim - 1) // 2
axis_meas = (0, 2)
xmax = args.diag_xmax
bins = args.diag_bins

# Create transforms
transfer_matrices = []
for i in range(ndim):
    for j in range(i):
        matrices = []
        for k, l in zip(axis_meas, (j, i)):
            matrix = np.identity(ndim)
            matrix[k, k] = matrix[l, l] = 0.0
            matrix[k, l] = matrix[l, k] = 1.0
            matrices.append(matrix)
        transfer_matrices.append(np.linalg.multi_dot(matrices[::-1]))

transforms = []
for matrix in transfer_matrices:
    transform = ment.LinearTransform(matrix)
    transforms.append(transform)

# Create diagnostics
edges = len(axis_meas) * [np.linspace(-xmax, xmax, bins + 1)]

diagnostics = []
for transform in transforms:
    diagnostic = ment.diag.HistogramND(
        axis=axis_meas,
        edges=edges, 
        blur=args.diag_blur,
        thresh=args.diag_thresh,
        thresh_type=args.diag_thresh_type,
    )
    diagnostics.append([diagnostic])

# Generate data
projections = ment.simulate_with_diag_update(
    x_true, transforms, diagnostics, blur=False, thresh=5.00e-03,
)

# Reconstruction model
# --------------------------------------------------------------------------------------

# Make sampler
samp_prop_cov = np.eye(ndim) * (args.samp_prop_scale ** 2)
samp_start = rng.multivariate_normal(np.zeros(ndim), np.eye(ndim), size=args.samp_chains)
samp_start = samp_start * args.samp_start_scale

sampler = ment.samp.MetropolisHastingsSampler(
    ndim=ndim,
    chains=args.samp_chains,
    proposal_cov=samp_prop_cov,
    start=samp_start,
    burnin=args.samp_burn,
    noise_scale=args.samp_noise,
    noise_type="gaussian",
    shuffle=True,
    verbose=True,
)

# Define Gaussian prior
prior = ment.GaussianPrior(ndim=ndim, scale=args.prior_scale)

# Create MENT model
model = ment.MENT(
    ndim=ndim,
    projections=projections,
    transforms=transforms,
    prior=prior,
    interpolation_kws=dict(method="linear"),
    sampler=sampler,
    nsamp=args.samp,
    mode="sample",
    verbose=True,
)


# Training
# --------------------------------------------------------------------------------------

eval_model = ment.train.Evaluator(args.samp)

trainer = ment.train.Trainer(
    model,
    plot_func=None,
    eval_func=eval_model,
    output_dir=output_dir,
)

trainer.train(epochs=args.epochs, learning_rate=args.lr)
