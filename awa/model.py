import copy
import math
from typing import Callable
from typing import Optional

import numpy as np
import torch

from bmadx import Particle
from bmadx.bmad_torch.track_torch import Beam
from bmadx.bmad_torch.track_torch import TorchLattice
from phase_space_reconstruction.virtual.beamlines import quad_tdc_bend


def coords_to_edges(coords: torch.Tensor) -> torch.Tensor:
    delta = coords[1] - coords[0]
    edges = torch.zeros(len(coords) + 1)
    edges[:-1] = coords - 0.5 * delta
    edges[-1] = coords[-1] + delta
    return edges


class WrappedNumpyTransform:
    def __init__(self, function: Callable) -> np.ndarray:
        self.function = function

    def __call__(self, x: np.ndarray, *args, **kwargs) -> np.ndarray:
        x_out = self.function(torch.from_numpy(x), *args, **kwargs)
        x_out = x_out.detach().numpy()
        return x_out


class TorchLatticeTransform(torch.nn.Module):
    def __init__(self, lattice: TorchLattice, beam_kws: dict = None) -> None:
        super().__init__()
        self.lattice = lattice

        self.beam_kws = beam_kws
        if self.beam_kws is None:
            self.beam_kws = dict()

        self.base_beam = Beam(torch.zeros((1, 6)), **self.beam_kws)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        beam = Beam(x, self.base_beam.p0c, self.base_beam.s, self.base_beam.mc2)
        lattice = copy.deepcopy(self.lattice)
        beam = lattice(beam)
        x_out = torch.vstack([beam.x, beam.px, beam.y, beam.py, beam.z, beam.pz]).T
        return x_out


class Factory:
    def __init__(self, params: torch.Tensor) -> None:
        self.params = params

    def dipole_on(self, index: int) -> bool:
        dipole_strength = self.params[index][2]
        dipole_on = dipole_strength > 1.00e-13
        return dipole_on


class LatticeFactory(Factory):
    def __init__(self, params: torch.Tensor, p0c: float = 43.3e06) -> None:
        super().__init__(params)
        self.p0c = torch.tensor(p0c)
        self.lattice0 = quad_tdc_bend(p0c=p0c, dipole_on=False)
        self.lattice1 = quad_tdc_bend(p0c=p0c, dipole_on=True)
        self.scan_ids = (0, 2, 4)
        self.beam_kws = dict(p0c=self.p0c)

    def make_lattice(self, index: int) -> TorchLattice:
        # https://github.com/roussel-ryan/gpsr_6d_paper/blob/main/phase_space_reconstruction/virtual/scans.py
        ids = self.scan_ids

        lattice = None
        if self.dipole_on(index):
            lattice = self.lattice1.copy()
            lattice.elements[ids[0]].K1.data = self.params[index][0]
            lattice.elements[ids[1]].VOLTAGE.data = self.params[index][1]

            G = self.params[index][2]
            l_bend = 0.3018
            theta = torch.arcsin(l_bend * G)  # AWA parameters
            l_arc = theta / G
            lattice.elements[ids[2]].G.data = G
            lattice.elements[ids[2]].L.data = l_arc
            lattice.elements[ids[2]].E2.data = theta
            lattice.elements[-1].L.data = 0.889 - l_bend / 2 / torch.cos(theta)

        else:
            lattice = self.lattice0.copy()
            lattice.elements[ids[0]].K1.data = self.params[index][0]
            lattice.elements[ids[1]].VOLTAGE.data = self.params[index][1]

            G = self.params[index][2]
            l_bend = 0.3018
            theta = torch.arcsin(l_bend * G)  # AWA parameters
            l_arc = theta / G
            lattice.elements[ids[2]].G.data = G
            lattice.elements[ids[2]].L.data = l_arc
            lattice.elements[ids[2]].E2.data = theta
            lattice.elements[-1].L.data = 0.889 - l_bend / 2 / torch.cos(theta)

        return lattice

    def make_transform(self, index: int) -> TorchLatticeTransform:
        lattice = self.make_lattice(index)
        transform = TorchLatticeTransform(lattice, beam_kws=self.beam_kws)
        return transform

    def make_transform_np(self, index: int) -> WrappedNumpyTransform:
        return WrappedNumpyTransform(self.make_transform(index))


class ImageFactory(Factory):
    def __init__(self, params: torch.Tensor, images: torch.Tensor) -> None:
        super().__init__(params)
        self.images = images
        self._screen0_size = 30.22e-03 * 300 / 700.0
        self._screen1_size = 26.96e-03 * 300 / 700.0

    def get_size(self, index: int) -> float:
        size = self._screen0_size
        if self.dipole_on(index):
            size = self._screen1_size
        return size

    def get_coords(self, index: int) -> list[torch.Tensor]:
        size = self.get_size(index)
        coords = [
            torch.linspace(-0.5 * size, 0.5 * size, self.images[index].shape[0]),
            torch.linspace(-0.5 * size, 0.5 * size, self.images[index].shape[1]),
        ]
        return coords

    def get_edges(self, index: int) -> torch.Tensor:
        return [coords_to_edges(c) for c in self.get_coords(index)]

    def get_values(self, index: int) -> torch.Tensor:
        coords = self.get_coords(index)
        values = torch.clone(self.images[index])
        values_sum = torch.sum(values)
        if values_sum > 0.0:
            values = values / values_sum
            values = values / math.prod(c[1] - c[0] for c in coords)
        return values

    def get_coords_np(self, index: int) -> list[np.ndarray]:
        return [c.numpy() for c in self.get_coords(index)]
    
    def get_values_np(self, index: int) -> np.ndarray:
        return self.get_values(index).numpy()
