import numpy as np
import torch

np.random.seed(2024)
torch.manual_seed(2024)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

L = 100

SW_SW = []
SW_MixSW = []
SW_SMixW = []

MixSW_SW = []
MixSW_MixSW = []
MixSW_SMixW = []

SMixW_SW = []
SMixW_MixSW = []
SMixW_SMixW = []

Y = np.load('saved/Y.npy')

Zs = np.load('saved/Zs.npy')
betas = np.load('saved/betas.npy')
mus = np.load('saved/mus.npy')
Sigmas = np.load('saved/Sigmas.npy')
weights = np.array([betas[i] * np.concatenate(([1], np.cumprod(1 - betas[i][:-1]))) for i in range(betas.shape[0])])
weights = weights / np.sum(weights, axis=1, keepdims=True)

mu2 = torch.from_numpy(mus).to(device).float()
Sigma2 = torch.from_numpy(Sigmas).to(device).float()
pi2 = torch.from_numpy(weights).to(device).float()

matrices_SW = np.load('saved/SW_L{}.npy'.format(L))
matrices_MixSW = np.load('saved/MixSW_L{}.npy'.format(L))
matrices_SMixW = np.load('saved/SMixW_L{}.npy'.format(L))

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

print('SW: SW: {}+-{}  - MixSW: {}+-{}   - SMixW: {}+-{} '
      .format(np.round(np.mean(SW_SW), 4), np.round(np.std(SW_SW), 4),
              np.round(np.mean(SW_MixSW), 4), np.round(np.std(SW_MixSW), 4),
              np.round(np.mean(SW_SMixW), 4), np.round(np.std(SW_SMixW), 4),
              )
      )

print('MixSW: SW: {}+-{}  - MixSW: {}+-{}  - SMixW: {}+-{} '
      .format(np.round(np.mean(MixSW_SW), 4), np.round(np.std(MixSW_SW), 4),
              np.round(np.mean(MixSW_MixSW), 4), np.round(np.std(MixSW_MixSW), 4),
              np.round(np.mean(MixSW_SMixW), 4), np.round(np.std(MixSW_SMixW), 4),
              )
      )

print('SMixW: SW: {}+-{} - MixSW: {}+-{}  - SMixW: {}+-{} '
      .format(np.round(np.mean(SMixW_SW), 4), np.round(np.std(SMixW_SW), 4),
              np.round(np.mean(SMixW_MixSW), 4), np.round(np.std(SMixW_MixSW), 4),
              np.round(np.mean(SMixW_SMixW), 4), np.round(np.std(SMixW_SMixW), 4),
              )
      )
