import ment
import torch
import zuko
from torch.utils.data import DataLoader
from typing import Callable


class DataNormalizer:
    def __init__(self, cov_matrix: torch.Tensor) -> None:
        self.cov_matrix = cov_matrix
        self.norm_matrix = ment.cov.normalization_matrix(cov_matrix.numpy(), scale=True)
        self.norm_matrix = torch.from_numpy(self.norm_matrix)
        self.norm_matrix = self.norm_matrix.float()
        self.unnorm_matrix = torch.linalg.inv(self.norm_matrix)
        
    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.unnorm_matrix.T)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.norm_matrix.T)
        

def make_flow(ndim: int = 6, transforms: int = 3, width: int = 64, depth: int = 3) -> zuko.flows.Flow:
    flow = zuko.flows.NSF(features=ndim, transforms=transforms, hidden_features=[width] * depth)
    return flow


def train_flow(
    flow: zuko.flows.Flow, 
    data_loader: DataLoader,
    data_normalizer: DataNormalizer = None,
    epochs: int = 10,
    lr: float = 0.0005, 
) -> dict:
    
    history = {}
    history["loss"] = []
    
    optimizer = torch.optim.Adam(flow.parameters(), lr=lr)        
    for epoch in range(epochs):
        for x in data_loader:
            if data_normalizer is not None:
                x = data_normalizer.normalize(x)
                
            loss = -torch.mean(flow().log_prob(x))
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            history["loss"].append(loss.detach())
        
    return history


def compute_flow_entropy(flow: zuko.flows.Flow, batch_size: int) -> torch.Tensor:
    with torch.no_grad():
        _, log_prob = flow().rsample_and_log_prob((batch_size,))
        return -torch.mean(log_prob)

