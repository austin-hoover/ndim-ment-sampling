import os
import pathlib
import numpy as np
import torch


path = pathlib.Path(__file__)
output_dir = os.path.join("outputs", path.stem)
os.makedirs(output_dir, exist_ok=True)

beam = torch.load("data/gpsr_paper_3d_scan_result.pt")
x_pred = torch.vstack([beam.x, beam.px, beam.y, beam.py, beam.z, beam.pz]).T

cov_matrix = torch.cov(x_pred.T)
np.savetxt(os.path.join(output_dir, "cov_matrix.dat"), cov_matrix.numpy())
