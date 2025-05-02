import numpy as np
import pandas as pd
import scipy.stats as sps
import tqdm
from sklearn.cluster import KMeans

np.random.seed(2024)


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


faithful_df = pd.read_csv('faithful.csv')[['eruptions', 'waiting']]
Y = faithful_df.values
np.save('saved/Y.npy', Y)
mu0 = np.array([3, 70])
Psi = np.array([[4, 0], [0, 36]])
lambda_ = 1
nu = 4
alpha = 1
K = 100
model = TDPGMM(alpha, mu0, lambda_, Psi, nu, d=2, K=K)
Zs, betas, mus, Sigmas = model.fit(Y, num_iters=10000)
Zs = Zs[-1000:]
betas = betas[-1000:]
mus = mus[-1000:]
Sigmas = Sigmas[-1000:]
np.save('saved/Zs.npy', np.array(Zs))
np.save('saved/betas.npy', np.array(betas))
np.save('saved/mus.npy', np.array(mus))
np.save('saved/Sigmas.npy', np.array(Sigmas))
np.savetxt('saved/Zs.txt', Zs, delimiter=',', fmt="%d")
