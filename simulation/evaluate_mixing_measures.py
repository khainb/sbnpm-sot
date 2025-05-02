import numpy as np
import torch
import tqdm

from libs.sot_gms import MixSW, SMixW
from libs.sw import SW

np.random.seed(2024)
torch.manual_seed(2024)
n = 200
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mus = np.array([[-2, -2], [2, -2], [-2, 2], [2, 2]])  # Means for the 4 components
sigmas = [1.5, 1.5, 1.5, 1.5]  # Standard deviations for each component

# Covariance matrices
covariances = np.array([np.diag([sigma ** 2, sigma ** 2]) for sigma in sigmas])
true_mu = torch.from_numpy(mus).float().to(device)
true_Sigma = torch.from_numpy(covariances).float().to(device)
true_pi = torch.Tensor([0.25, 0.25, 0.25, 0.25]).float().to(device)

repetitions = 25
n = 200
L = 100
K = 100

SW_SW = []
SW_SWtrue = []
SW_MixSW = []
SW_MixSWtrue = []
SW_SMixW = []
SW_SMixWtrue = []

MixSW_SW = []
MixSW_SWtrue = []
MixSW_MixSW = []
MixSW_MixSWtrue = []
MixSW_SMixW = []
MixSW_SMixWtrue = []

SMixW_SW = []
SMixW_SWtrue = []
SMixW_MixSW = []
SMixW_MixSWtrue = []
SMixW_SMixW = []
SMixW_SMixWtrue = []

for time in tqdm.tqdm(range(repetitions)):
    Y = np.load('saved/Y_n{}_repeat{}.npy'.format(n, time))
    labels = np.load('saved/label_n{}_repeat{}.npy'.format(n, time))

    Zs = np.load('saved/Zs_n{}_K{}_repeat{}.npy'.format(n, K, time))
    betas = np.load('saved/betas_n{}_K{}_repeat{}.npy'.format(n, K, time))
    mus = np.load('saved/mus_n{}_K{}_repeat{}.npy'.format(n, K, time))
    Sigmas = np.load('saved/Sigmas_n{}_K{}_repeat{}.npy'.format(n, K, time))
    weights = np.array([betas[i] * np.concatenate(([1], np.cumprod(1 - betas[i][:-1]))) for i in range(betas.shape[0])])
    weights = weights / np.sum(weights, axis=1, keepdims=True)

    mu2 = torch.from_numpy(mus).to(device).float()
    Sigma2 = torch.from_numpy(Sigmas).to(device).float()
    pi2 = torch.from_numpy(weights).to(device).float()

    matrices_SW = np.load('saved/SW_n{}_L{}_K{}_repeat{}.npy'.format(n, L, K, time))
    matrices_MixSW = np.load('saved/MixSW_n{}_L{}_K{}_repeat{}.npy'.format(n, L, K, time))
    matrices_SMixW = np.load('saved/SMixW_n{}_L{}_K{}_repeat{}.npy'.format(n, L, K, time))

    indx_sw = np.argmin(np.mean(matrices_SW, axis=0))
    indx_mixsw = np.argmin(np.mean(matrices_MixSW, axis=0))
    indx_smixsw = np.argmin(np.mean(matrices_SMixW, axis=0))

    SW_SW.append(np.round(np.mean(matrices_SW[indx_sw]), 4))
    SW_MixSW.append(np.round(np.mean(matrices_MixSW[indx_sw]), 4))
    SW_SMixW.append(np.round(np.mean(matrices_SMixW[indx_sw]), 4))

    MixSW_SW.append(np.round(np.mean(matrices_SW[indx_mixsw]), 4))
    MixSW_MixSW.append(np.round(np.mean(matrices_MixSW[indx_mixsw]), 4))
    MixSW_SMixW.append(np.round(np.mean(matrices_SMixW[indx_mixsw]), 4))

    SMixW_SW.append(np.round(np.mean(matrices_SW[indx_smixsw]), 4))
    SMixW_MixSW.append(np.round(np.mean(matrices_MixSW[indx_smixsw]), 4))
    SMixW_SMixW.append(np.round(np.mean(matrices_SMixW[indx_smixsw]), 4))

    torch.manual_seed(2024)
    SW_SWtrue.append(SW(torch.cat([mu2[indx_sw], Sigma2[indx_sw].view(Sigma2[indx_sw].shape[0], -1)], dim=1),
                        torch.cat([true_mu, true_Sigma.view(true_Sigma.shape[0], -1)], dim=1), pi2[indx_sw], true_pi,
                        L=1000).cpu().detach().numpy())
    torch.manual_seed(2024)
    MixSW_SWtrue.append(
        SW(torch.cat([mu2[indx_mixsw], Sigma2[indx_mixsw].view(Sigma2[indx_mixsw].shape[0], -1)], dim=1),
           torch.cat([true_mu, true_Sigma.view(true_Sigma.shape[0], -1)], dim=1), pi2[indx_mixsw], true_pi,
           L=1000).cpu().detach().numpy())
    torch.manual_seed(2024)
    SMixW_SWtrue.append(
        SW(torch.cat([mu2[indx_smixsw], Sigma2[indx_smixsw].view(Sigma2[indx_smixsw].shape[0], -1)], dim=1),
           torch.cat([true_mu, true_Sigma.view(true_Sigma.shape[0], -1)], dim=1), pi2[indx_smixsw], true_pi,
           L=1000).cpu().detach().numpy())

    torch.manual_seed(2024)
    SW_MixSWtrue.append(
        MixSW(mu2[indx_sw], Sigma2[indx_sw], true_mu, true_Sigma, pi2[indx_sw], true_pi, L=1000).cpu().detach().numpy())
    torch.manual_seed(2024)
    MixSW_MixSWtrue.append(MixSW(mu2[indx_mixsw], Sigma2[indx_mixsw], true_mu, true_Sigma, pi2[indx_mixsw], true_pi,
                                 L=1000).cpu().detach().numpy())
    torch.manual_seed(2024)
    SMixW_MixSWtrue.append(MixSW(mu2[indx_smixsw], Sigma2[indx_smixsw], true_mu, true_Sigma, pi2[indx_smixsw], true_pi,
                                 L=1000).cpu().detach().numpy())

    torch.manual_seed(2024)
    SW_SMixWtrue.append(
        SMixW(mu2[indx_sw], Sigma2[indx_sw], true_mu, true_Sigma, pi2[indx_sw], true_pi, L=1000).cpu().detach().numpy())
    torch.manual_seed(2024)
    MixSW_SMixWtrue.append(SMixW(mu2[indx_mixsw], Sigma2[indx_mixsw], true_mu, true_Sigma, pi2[indx_mixsw], true_pi,
                                 L=1000).cpu().detach().numpy())
    torch.manual_seed(2024)
    SMixW_SMixWtrue.append(SMixW(mu2[indx_smixsw], Sigma2[indx_smixsw], true_mu, true_Sigma, pi2[indx_smixsw], true_pi,
                                 L=1000).cpu().detach().numpy())

print('SW: SW: {}+-{} - SWtrue: {}+-{}  - MixSW: {}+-{}  - MixSWtrue: {}+-{} - SMixW: {}+-{} - SMixWtrue: {}+-{}'
      .format(np.round(np.mean(SW_SW), 4), np.round(np.std(SW_SW), 4),
              np.round(np.mean(SW_SWtrue), 4), np.round(np.std(SW_SWtrue), 4),
              np.round(np.mean(SW_MixSW), 4), np.round(np.std(SW_MixSW), 4),
              np.round(np.mean(SW_MixSWtrue), 4), np.round(np.std(SW_MixSWtrue), 4),
              np.round(np.mean(SW_SMixW), 4), np.round(np.std(SW_SMixW), 4),
              np.round(np.mean(SW_SMixWtrue), 4), np.round(np.std(SW_SMixWtrue), 4)
              )
      )

print('MixSW: SW: {}+-{} - SWtrue: {}+-{}  - MixSW: {}+-{}  - MixSWtrue: {}+-{} - SMixW: {}+-{} - SMixWtrue: {}+-{}'
      .format(np.round(np.mean(MixSW_SW), 4), np.round(np.std(MixSW_SW), 4),
              np.round(np.mean(MixSW_SWtrue), 4), np.round(np.std(MixSW_SWtrue), 4),
              np.round(np.mean(MixSW_MixSW), 4), np.round(np.std(MixSW_MixSW), 4),
              np.round(np.mean(MixSW_MixSWtrue), 4), np.round(np.std(MixSW_MixSWtrue), 4),
              np.round(np.mean(MixSW_SMixW), 4), np.round(np.std(MixSW_SMixW), 4),
              np.round(np.mean(MixSW_SMixWtrue), 4), np.round(np.std(MixSW_SMixWtrue), 4)
              )
      )

print('SMixW: SW: {}+-{} - SWtrue: {}+-{}  - MixSW: {}+-{}  - MixSWtrue: {}+-{} - SMixW: {}+-{} - SMixWtrue: {}+-{}'
      .format(np.round(np.mean(SMixW_SW), 4), np.round(np.std(SMixW_SW), 4),
              np.round(np.mean(SMixW_SWtrue), 4), np.round(np.std(SMixW_SWtrue), 4),
              np.round(np.mean(SMixW_MixSW), 4), np.round(np.std(SMixW_MixSW), 4),
              np.round(np.mean(SMixW_MixSWtrue), 4), np.round(np.std(SMixW_MixSWtrue), 4),
              np.round(np.mean(SMixW_SMixW), 4), np.round(np.std(SMixW_SMixW), 4),
              np.round(np.mean(SMixW_SMixWtrue), 4), np.round(np.std(SMixW_SMixWtrue), 4)
              )
      )
