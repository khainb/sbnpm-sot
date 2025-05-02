import numpy as np
import scipy.stats as sps
import torch
import tqdm

from libs.sot_gms import SMixW
from libs.sw import SW

true_mus = np.array([[-2, -2], [2, -2], [-2, 2], [2, 2]])  # Means for the 4 components
sigmas = [1.5, 1.5, 1.5, 1.5]  # Standard deviations for each component
true_weight = np.array([0.25, 0.25, 0.25, 0.25])
# Covariance matrices
true_Sigmas = [np.diag([sigma ** 2, sigma ** 2]) for sigma in sigmas]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(2024)
torch.manual_seed(2024)
repetitions = 25
n = 200
L = 100
for time in range(repetitions):
    np.random.seed(2024)
    torch.manual_seed(2024)
    Y = np.load('saved/Y_n{}_repeat{}.npy'.format(n, time))
    labels = np.load('saved/label_n{}_repeat{}.npy'.format(n, time))
    for K in [100]:
        Zs = np.load('saved/Zs_n{}_K{}_repeat{}.npy'.format(n, K, time))
        betas = np.load('saved/betas_n{}_K{}_repeat{}.npy'.format(n, K, time))
        mus = np.load('saved/mus_n{}_K{}_repeat{}.npy'.format(n, K, time))
        Sigmas = np.load('saved/Sigmas_n{}_K{}_repeat{}.npy'.format(n, K, time))
        weights = np.array(
            [betas[i] * np.concatenate(([1], np.cumprod(1 - betas[i][:-1]))) for i in range(betas.shape[0])])
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
        for i in tqdm.tqdm(range(N)):
            for j in range(i + 1, N):
                with torch.no_grad():
                    matrices[i, j] = SMixW(mu2[i], Sigma2[i], mu2[j], Sigma2[j], pi2[i], pi2[j], L=L)
                    if (np.isnan(matrices[i, j])):
                        print('Nan')
                    matrices[j, i] = matrices[i, j]

        np.save('saved/SMixW_n{}_L{}_K{}_repeat{}.npy'.format(n, L, K, time), matrices)
        indx_sot = np.argmin(np.mean(matrices, axis=0))

        beta, mu, Sigma = betas[indx_sot], mus[indx_sot], Sigmas[indx_sot]

        weight = beta * np.concatenate(([1], np.cumprod(1 - beta[:-1])))
        fYs = np.stack([sps.multivariate_normal.pdf(Y, mean=mu[k], cov=Sigma[k]) for k in range(mu.shape[0])], axis=1)
        pi = weight * fYs
        Z = np.argmax(pi, axis=1)
        np.savetxt('saved/Zs_SMixW_n{}_L{}_K{}_repeat{}.txt'.format(n, L, K, time), Z.reshape(1, -1), delimiter=',',
                   fmt="%d")
        print(np.unique(Z))
        num = 100
        data = Y
        alpha = np.linspace(np.min(data[:, 0]) - 1, np.max(data[:, 0]) + 1, num)
        beta = np.linspace(np.min(data[:, 1]) - 1, np.max(data[:, 1]) + 1, num)
        xx, yy = np.meshgrid(alpha, beta)
        points = np.stack((xx, yy), axis=-1)
        xgrid = points.reshape(-1, 2)
        true_density = fbar(xgrid, true_weight, true_mus, true_Sigmas)
        true_density = true_density / np.sum(true_density)
        xgrid = points.reshape(-1, 2)
        SOT_density = fbar(xgrid, betas[indx_sot], mus[indx_sot], Sigmas[indx_sot])
        SOT_density = SOT_density / np.sum(SOT_density)

        W2 = []
        TV = []
        trueTV = []
        trueW2 = []
        xgrid_cuda = torch.from_numpy(xgrid).float().to(device)
        SOT_density_cuda = torch.from_numpy(SOT_density).float().to(device)
        true_density_cuda = torch.from_numpy(true_density).float().to(device)
        for i in tqdm.tqdm(range(len(mus))):
            density = fbar(xgrid, betas[i], mus[i], Sigmas[i])
            density = density / np.sum(density)
            TV.append(np.sum(np.abs(SOT_density - density)))
            W2.append(SW(xgrid_cuda, xgrid_cuda, SOT_density_cuda, torch.from_numpy(density).float().to(device),
                         L=1000).detach().cpu().numpy())
            trueTV.append(np.sum(np.abs(SOT_density - true_density)))
            trueW2.append(SW(xgrid_cuda, xgrid_cuda, SOT_density_cuda, true_density_cuda,
                             L=1000).detach().cpu().numpy())

        np.save('saved/SMixW_TV_n{}_L{}_K{}_repeat{}.npy'.format(n, L, K, time), np.array(TV))
        np.save('saved/SMixW_SW_n{}_L{}_K{}_repeat{}.npy'.format(n, L, K, time), np.array(W2))
        np.save('saved/SMixW_trueTV_n{}_L{}_K{}_repeat{}.npy'.format(n, L, K, time), np.array(trueTV))
        np.save('saved/SMixW_trueSW_n{}_L{}_K{}_repeat{}.npy'.format(n, L, K, time), np.array(trueW2))
