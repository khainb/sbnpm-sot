import matplotlib.pyplot as plt
import numpy as np
import torch

from libs.sot_gms import MixSW, SMixW
from libs.sw import SW

np.random.seed(2024)
torch.manual_seed(2024)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# GMM parameters
weights1 = torch.tensor([0.3, 0.4, 0.3], device=device)
means1 = torch.tensor([[0., 0.], [3., 3.], [-3., 3.]], device=device)
covs1 = torch.stack([
    torch.tensor([[1.0, 0.8], [0.8, 1.5]]),
    torch.tensor([[0.7, -0.3], [-0.3, 0.9]]),
    torch.tensor([[2.0, 1.0], [1.0, 3.0]])
]).to(device)

weights2 = torch.tensor([0.2, 0.5, 0.3], device=device)
means2 = torch.tensor([[1., 1.], [4., 4.], [-2., 2.]], device=device)
covs2 = torch.stack([
    torch.tensor([[1.2, 0.5], [0.5, 0.8]]),
    torch.tensor([[0.4, -0.2], [-0.2, 0.6]]),
    torch.tensor([[1.5, 0.7], [0.7, 2.5]])
]).to(device)

# Range of projections
L_vals = np.arange(1, 1002, 10)
MixSW_vals = []
SMixW_vals = []
SW_vals = []

# Evaluate both distances
for L in L_vals:
    MixSW_vals.append(MixSW(means1, covs1, means2, covs2, weights1, weights2, L=L).item())
    SMixW_vals.append(SMixW(means1, covs1, means2, covs2, weights1, weights2, L=L).item())
    SW_vals.append(SW(
        torch.cat([means1, covs1.view(covs1.shape[0], -1)], dim=1),
        torch.cat([means2, covs2.view(covs2.shape[0], -1)], dim=1),
        weights1, weights2, L=L).item())

# Merged plot with line styles
plt.figure(figsize=(8, 5))
plt.plot(L_vals, SW_vals, label='SW', marker='d', linestyle='-', color='tab:red', markersize=5)
plt.plot(L_vals, MixSW_vals, label='Mix-SW', marker='o', linestyle='--', color='tab:green', markersize=5)
plt.plot(L_vals, SMixW_vals, label='SMix-W', marker='s', linestyle='-.', color='tab:blue', markersize=5)
plt.xlabel('Number of projections (L)', fontsize=25)
plt.ylabel('Approximation Value', fontsize=25)
plt.title('Monte Carlo Approximation', fontsize=25)
plt.legend(fontsize=25)
plt.grid(True)
plt.tight_layout()
plt.show()
