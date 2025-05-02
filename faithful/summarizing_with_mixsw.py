import numpy as np
import scipy.stats as sps
import torch
import tqdm

from libs.sot_gms import MixSW
from libs.sw import SW

np.random.seed(2024)
torch.manual_seed(2024)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

L = 100
np.random.seed(2024)
torch.manual_seed(2024)
Y = np.load('saved/Y.npy')

Zs = np.load('saved/Zs.npy')
betas = np.load('saved/betas.npy')
mus = np.load('saved/mus.npy')
Sigmas = np.load('saved/Sigmas.npy')
weights = np.array([betas[i] * np.concatenate(([1], np.cumprod(1 - betas[i][:-1]))) for i in range(betas.shape[0])])
weights = weights / np.sum(weights, axis=1, keepdims=True)


def fbar(xgrid, betas, th, S):
    k = th.shape[0]
    w = betas * np.concatenate(([1], np.cumprod(1 - betas[:-1])))
    fx = 0
    for j in range(k):
        fx = fx + w[j] * sps.multivariate_normal.pdf(xgrid, mean=th[j], cov=S[j])
    return fx


mu2 = torch.from_numpy(mus).to(device).float()
Sigma2 = torch.from_numpy(Sigmas).to(device).float()
pi2 = torch.from_numpy(weights).to(device).float()
N = mus.shape[0]
#
matrices = np.zeros((N, N))

#
for i in tqdm.tqdm(range(N)):
    for j in range(i + 1, N):
        with torch.no_grad():
            matrices[i, j] = MixSW(mu2[i], Sigma2[i], mu2[j], Sigma2[j], pi2[i], pi2[j], L=L)
            if (np.isnan(matrices[i, j])):
                print('Nan')
            matrices[j, i] = matrices[i, j]

np.save('saved/MixSW_L{}.npy'.format(L), matrices)
indx_sot = np.argmin(np.mean(matrices, axis=0))

beta, mu, Sigma = betas[indx_sot], mus[indx_sot], Sigmas[indx_sot]

weight = beta * np.concatenate(([1], np.cumprod(1 - beta[:-1])))
fYs = np.stack([sps.multivariate_normal.pdf(Y, mean=mu[k], cov=Sigma[k]) for k in range(mu.shape[0])], axis=1)
pi = weight * fYs
Z = np.argmax(pi, axis=1)
np.savetxt('saved/Zs_MixSW_L{}.txt'.format(L), Z.reshape(1, -1), delimiter=',', fmt="%d")

num = 100
data = Y
alpha = np.linspace(np.min(data[:, 0]) - 1, np.max(data[:, 0]) + 1, num)
beta = np.linspace(np.min(data[:, 1]) - 1, np.max(data[:, 1]) + 1, num)
xx, yy = np.meshgrid(alpha, beta)
points = np.stack((xx, yy), axis=-1)
xgrid = points.reshape(-1, 2)

SOT_density = fbar(xgrid, betas[indx_sot], mus[indx_sot], Sigmas[indx_sot])
SOT_density = SOT_density / np.sum(SOT_density)

W2 = []
TV = []
xgrid_cuda = torch.from_numpy(xgrid).float().to(device)
SOT_density_cuda = torch.from_numpy(SOT_density).float().to(device)
for i in tqdm.tqdm(range(len(mus))):
    density = fbar(xgrid, betas[i], mus[i], Sigmas[i])
    density = density / np.sum(density)
    TV.append(np.sum(np.abs(SOT_density - density)))
    W2.append(SW(xgrid_cuda, xgrid_cuda, SOT_density_cuda, torch.from_numpy(density).float().to(device),
                 L=1000).detach().cpu().numpy())

np.save('saved/MixSW_TV_L{}.npy'.format(L), np.array(TV))
np.save('saved/MixSW_SW_L{}.npy'.format(L), np.array(W2))
