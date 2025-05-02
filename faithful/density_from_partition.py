import numpy as np
import scipy.stats as sps
from sklearn.cluster import KMeans
import tqdm
import torch
from libs.sw import SW

np.random.seed(2024)
torch.manual_seed(2024)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TDPGMM:
    def __init__(self, alpha, mu0, lambda_, Psi, nu, d, K):
        self.alpha = alpha
        self.mu0 = mu0
        self.lambda_ = lambda_
        self.Psi = Psi
        self.nu = nu
        self.K = K
        self.d = d

    def sample_normal_inverse_wishart(self, mulocal, lambda_local, psilocal, nulocal):
        """
        Sample from the Normal-Inverse-Wishart distribution.

        Parameters:
        - mu_0: Mean vector of the normal distribution (shape: d)
        - lambda_: Scale parameter (scalar)
        - psi: Scale matrix for the Wishart distribution (shape: d x d)
        - nu: Degrees of freedom for the Wishart distribution (scalar, should be > d-1)
        - size: Number of samples to draw

        Returns:
        - mu_samples: Samples from the normal component (shape: size x d)
        - sigma_samples: Samples from the inverse Wishart component (shape: size x d x d)
        """

        # Sample from the inverse Wishart distribution
        sigma_samples = sps.invwishart.rvs(df=nulocal, scale=psilocal, size=1)
        # sigma_samples = np.linalg.inv(sigma_samples)  # Inverse for the Inverse-Wishart
        # Sample from the normal distribution
        mu_samples = np.random.multivariate_normal(mulocal, sigma_samples / lambda_local, size=1)

        return mu_samples, sigma_samples

    def sample_z(self, Y, mus, Sigmas, betas):
        weights = betas * np.concatenate(([1], np.cumprod(1 - betas[:-1])))
        fYs = np.stack([sps.multivariate_normal.pdf(Y, mean=mus[k], cov=Sigmas[k]) for k in range(self.K)], axis=1)
        pi = weights * fYs
        pi = pi / np.sum(pi, axis=1, keepdims=True)
        z = np.array([np.random.choice(self.K, 1, p=pi[i])[0] for i in range(pi.shape[0])])
        return z

    def sample_beta(self, z):
        n_k = np.zeros(self.K)
        for k in range(self.K):
            n_k[k] = np.sum(z == k)
        betas = np.zeros(self.K)
        for k in range(self.K):
            betas[k] = np.random.beta(1 + n_k[k], self.alpha + np.sum(n_k[k + 1:]))
        return betas

    def sample_mus_Sigmas(self, Y, z):
        mus = []
        Sigmas = []
        for k in range(self.K):
            inds = z == k
            n_k = np.sum(inds)
            if n_k != 0:
                Ybar = np.mean(Y[inds], axis=0)
                muprime = (self.lambda_ * self.mu0 + n_k * Ybar) / (self.lambda_ + n_k)
                Psiprime = self.Psi + np.matmul((Y[inds] - Ybar).T, (Y[inds] - Ybar)) + (self.lambda_ * n_k) / (
                        self.lambda_ + n_k) * np.matmul((Ybar - self.mu0).reshape(1, -1).T,
                                                        (Ybar - self.mu0).reshape(1, -1))
                lambdaprime = self.lambda_ + n_k
                nuprime = self.nu + n_k
            else:
                muprime = self.mu0
                Psiprime = self.Psi
                lambdaprime = self.lambda_
                nuprime = self.nu
            sampled_mu, sampled_Sigma = self.sample_normal_inverse_wishart(muprime, lambdaprime, Psiprime, nuprime)
            mus.append(sampled_mu[0])
            Sigmas.append(sampled_Sigma)
        return np.array(mus), np.array(Sigmas)

    def fit(self, Y, num_iters=1000):
        clustering = KMeans(n_clusters=10).fit(Y)
        z = np.array(clustering.labels_)
        Zs = []
        betas = []
        mus = []
        Sigmas = []
        for i in tqdm.tqdm(range(num_iters)):
            beta = self.sample_beta(z)
            mu, Sigma = self.sample_mus_Sigmas(Y, z)
            z = self.sample_z(Y, mu, Sigma, beta)
            Zs.append(z)
            betas.append(beta)
            mus.append(mu)
            Sigmas.append(Sigma)
        print(np.unique(Zs[-1]))
        self.results = {'Zs': Zs, 'betas': betas, 'mus': mus, 'Sigmas': Sigmas}
        return Zs, betas, mus, Sigmas


mu0 = np.array([3, 70])
Psi = np.array([[4, 0], [0, 36]])
lambda_ = 1
nu = 4
alpha = 1

for name in ['Binder', 'VI', 'omARI']:
    all_TV = []
    all_SW = []
    model = TDPGMM(alpha, mu0, lambda_, Psi, nu, d=2, K=100)
    np.random.seed(2024)
    torch.manual_seed(2024)
    Y = np.load('saved/Y.npy')
    Zs = np.loadtxt('saved/{}.csv'.format(name))
    pos_betas = np.load('saved/betas.npy')
    pos_mus = np.load('saved/mus.npy')
    pos_Sigmas = np.load('saved/Sigmas.npy')
    pos_weights = np.array(
        [pos_betas[i] * np.concatenate(([1], np.cumprod(1 - pos_betas[i][:-1]))) for i in range(pos_betas.shape[0])])
    pos_weights = pos_weights / np.sum(pos_weights, axis=1, keepdims=True)


    def fbar(xgrid, betas, th, S):
        k = th.shape[0]
        w = betas * np.concatenate(([1], np.cumprod(1 - betas[:-1])))
        fx = 0
        for j in range(k):
            fx = fx + w[j] * sps.multivariate_normal.pdf(xgrid, mean=th[j], cov=S[j])
        return fx


    num = 100
    data = Y
    alpha_grid = np.linspace(np.min(data[:, 0]) - 1, np.max(data[:, 0]) + 1, num)
    beta_grid = np.linspace(np.min(data[:, 1]) - 1, np.max(data[:, 1]) + 1, num)
    xx, yy = np.meshgrid(alpha_grid, beta_grid)
    points = np.stack((xx, yy), axis=-1)
    xgrid = points.reshape(-1, 2)

    W2 = []
    TV = []
    densities = []
    xgrid_cuda = torch.from_numpy(xgrid).float().to(device)
    for _ in tqdm.tqdm(range(10)):
        betas = model.sample_beta(Zs)
        mus, Sigmas = model.sample_mus_Sigmas(Y, Zs)

        SOT_density = fbar(xgrid, betas, mus, Sigmas)
        densities.append(SOT_density)
        SOT_density = SOT_density / np.sum(SOT_density)

        SOT_density_cuda = torch.from_numpy(SOT_density).float().to(device)
        for i in range(len(pos_mus)):
            density = fbar(xgrid, pos_betas[i], pos_mus[i], pos_Sigmas[i])
            density = density / np.sum(density)
            TV.append(np.sum(np.abs(SOT_density - density)))
            W2.append(SW(xgrid_cuda, xgrid_cuda, SOT_density_cuda, torch.from_numpy(density).float().to(device),
                         L=1000).detach().cpu().numpy())
    all_TV.append(np.mean(TV))
    all_SW.append(np.mean(W2))
    np.save('saved/{}_TV.npy'.format(name), np.array(all_TV))
    np.save('saved/{}_SW.npy'.format(name), np.array(all_SW))
