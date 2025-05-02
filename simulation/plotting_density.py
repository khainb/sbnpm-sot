import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sps
import torch
import tqdm
from sklearn.cluster import KMeans

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


mu0 = np.zeros(2)
Psi = np.array([[1, 0], [0, 1]])
lambda_ = 1
nu = 4

alpha = 1

true_mus = np.array([[-2, -2], [2, -2], [-2, 2], [2, 2]])  # Means for the 4 components
sigmas = [1.5, 1.5, 1.5, 1.5]  # Standard deviations for each component
true_weight = np.array([0.25, 0.25, 0.25, 0.25])
# Covariance matrices
true_Sigmas = [np.diag([sigma ** 2, sigma ** 2]) for sigma in sigmas]
n = 200
K = 100

time = 0

Y = np.load('saved/Y_n{}_repeat{}.npy'.format(n, time))
label = np.load('saved/label_n{}_repeat{}.npy'.format(n, time))
pos_betas = np.load('saved/betas_n{}_K{}_repeat{}.npy'.format(n, K, time))
pos_mus = np.load('saved/mus_n{}_K{}_repeat{}.npy'.format(n, K, time))
pos_Sigmas = np.load('saved/Sigmas_n{}_K{}_repeat{}.npy'.format(n, K, time))
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
plt.figure(figsize=(6, 6))

alpha_grid = np.linspace(np.min(Y[:, 0]) - 1, np.max(Y[:, 0]) + 1, num)
beta_grid = np.linspace(np.min(Y[:, 1]) - 1, np.max(Y[:, 1]) + 1, num)
xx, yy = np.meshgrid(alpha_grid, beta_grid)
points = np.stack((xx, yy), axis=-1)
xgrid = points.reshape(-1, 2)

# Define markers
markers = ['o', 's', '^', 'D', 'v', '*', 'P', 'X', '<', '>']
fixed_colors = [
    '#1f77b4',  # blue
    '#ff7f0e',  # orange
    '#2ca02c',  # green
    '#d62728',  # red
    '#9467bd',  # purple
    '#8c564b',  # brown
    '#e377c2',  # pink
    '#7f7f7f',  # gray
    '#bcbd22',  # olive
    '#17becf'  # cyan
]
# Only for non-empty clusters
unique_labels = np.unique(label)
non_empty_clusters = [lab for lab in unique_labels if np.sum(label == lab) > 0]

# Assert enough markers
assert len(non_empty_clusters) <= len(markers), (
    f"Not enough markers: {len(non_empty_clusters)} clusters but only {len(markers)} markers. "
    f"Add more markers!"
)

# Plot points cluster by cluster
marker_idx = 0
for lab in non_empty_clusters:
    mask = label == lab
    marker = markers[marker_idx % len(markers)]
    plt.scatter(Y[mask, 0], Y[mask, 1],
                alpha=0.6,
                c=fixed_colors[lab],
                marker=marker,
                edgecolor='k',
                label=f'{lab + 1}')
    marker_idx += 1

# Plot true density contours
density_values = np.mean([sps.multivariate_normal.pdf(xgrid, mean=true_mus[j], cov=true_Sigmas[j]) for j in range(4)],
                         axis=0)
plt.contour(alpha_grid, beta_grid, density_values.reshape(num, num), levels=20)

plt.title("Simulation True", fontsize=25)
plt.xlabel("$y^{(1)}$", fontsize=25)
plt.ylabel("$y^{(2)}$", fontsize=25)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 6))

fbars = [fbar(xgrid, pos_betas[i], pos_mus[i], pos_Sigmas[i]) for i in range(len(pos_mus))]
plt.scatter(Y[:, 0], Y[:, 1], alpha=0.6, cmap='viridis', edgecolor='k')
plt.contour(alpha_grid, beta_grid, np.mean(fbars, axis=0).reshape(num, num), levels=20)
plt.title("Posterior Mean".format(n), fontsize=25)
plt.xlabel("$y^{(1)}$", fontsize=25)
plt.ylabel("$y^{(2)}$", fontsize=25)
plt.tight_layout()
# plt.legend()
plt.show()

for name in ['Binder', 'VI', 'omARI']:
    Zs = np.loadtxt('saved/{}_n200_rep{}.csv'.format(name, time)).astype(np.int32)

    model = TDPGMM(alpha, mu0, lambda_, Psi, nu, d=2, K=K)


    def fbar(xgrid, betas, th, S, weight=False):
        k = th.shape[0]
        if (weight):
            w = betas
        else:
            w = betas * np.concatenate(([1], np.cumprod(1 - betas[:-1])))
        fx = 0
        for j in range(k):
            fx = fx + w[j] * sps.multivariate_normal.pdf(xgrid, mean=th[j], cov=S[j])
        return fx


    num = 100
    true_density = fbar(xgrid, true_weight, true_mus, true_Sigmas)
    true_density = true_density / np.sum(true_density)

    W2 = []
    TV = []
    densities = []
    trueTV = []
    trueW2 = []
    xgrid_cuda = torch.from_numpy(xgrid).float().to(device)
    true_density_cuda = torch.from_numpy(true_density).float().to(device)
    for _ in tqdm.tqdm(range(10)):
        betas = model.sample_beta(Zs)
        mus_sample, Sigmas_sample = model.sample_mus_Sigmas(Y, Zs)

        SOT_density = fbar(xgrid, betas, mus_sample, Sigmas_sample)
        densities.append(SOT_density)

    plt.figure(figsize=(6, 6))

    # Define markers
    markers = ['o', 's', '^', 'D', 'v', '*', 'P', 'X', '<', '>']

    # Only for non-empty clusters
    unique_Zs = np.unique(Zs)
    non_empty_clusters = [z_val for z_val in unique_Zs if np.sum(Zs == z_val) > 0]

    # Assert enough markers
    assert len(non_empty_clusters) <= len(markers), (
        f"Not enough markers: {len(non_empty_clusters)} clusters but only {len(markers)} markers. "
        f"Add more markers!"
    )

    marker_idx = 0

    for z_val in non_empty_clusters:
        mask = Zs == z_val
        marker = markers[(marker_idx % len(markers))]
        plt.scatter(Y[mask, 0], Y[mask, 1],
                    alpha=0.6,
                    c=fixed_colors[z_val - 1],
                    marker=marker,
                    edgecolor='k',
                    label=f'{z_val}')
        marker_idx += 1

    # Plot density contours
    plt.contour(alpha_grid, beta_grid, np.mean(densities, axis=0).reshape(num, num), levels=20)

    plt.title('{}'.format(name), fontsize=25)
    plt.xlabel("$y^{(1)}$", fontsize=25)
    plt.ylabel("$y^{(2)}$", fontsize=25)
    plt.legend()
    plt.tight_layout()
    plt.show()

L = 100
for name in ['SW', 'MixSW', 'SMixW']:
    matrices = np.load('saved/{}_n{}_L{}_K{}_repeat{}.npy'.format(name, n, L, K, time))
    # matrices=np.load('saved/mixSW_{}_{}.npy'.format(L,n))
    indx_sot = np.argmin(np.mean(matrices, axis=0))
    Zs = np.loadtxt('saved/Zs_{}_n{}_L{}_K{}_repeat{}.txt'.format(name, n, L, K, time), delimiter=',')
    unique_ids, Zs = np.unique(Zs, return_inverse=True)
    betas = np.load('saved/betas_n{}_K{}_repeat{}.npy'.format(n, K, time))
    mus = np.load('saved/mus_n{}_K{}_repeat{}.npy'.format(n, K, time))
    Sigmas = np.load('saved/Sigmas_n{}_K{}_repeat{}.npy'.format(n, K, time))


    def fbar(xgrid, betas, th, S, weight=False):
        k = th.shape[0]
        if (weight):
            w = betas
        else:
            w = betas * np.concatenate(([1], np.cumprod(1 - betas[:-1])))
        fx = 0
        for j in range(k):
            fx = fx + w[j] * sps.multivariate_normal.pdf(xgrid, mean=th[j], cov=S[j])
        return fx


    plt.figure(figsize=(6, 6))

    # Define markers
    markers = ['o', 's', '^', 'D', 'v', '*', 'P', 'X', '<', '>']
    label = Zs
    # Only for non-empty clusters
    unique_labels = np.unique(label)
    non_empty_clusters = [lab for lab in unique_labels if np.sum(label == lab) > 0]

    # Assert enough markers
    assert len(non_empty_clusters) <= len(markers), (
        f"Not enough markers: {len(non_empty_clusters)} clusters but only {len(markers)} markers. "
        f"Add more markers!"
    )

    # Plot points cluster by cluster
    marker_idx = 0
    for lab in non_empty_clusters:
        mask = label == lab
        marker = markers[marker_idx % len(markers)]
        plt.scatter(Y[mask, 0], Y[mask, 1],
                    alpha=0.6,
                    c=fixed_colors[int(lab)],
                    marker=marker,
                    edgecolor='k',
                    label=f'{int(lab + 1)}')
        marker_idx += 1

    # Plot true density contours
    SOT_density = fbar(xgrid, betas[indx_sot], mus[indx_sot], Sigmas[indx_sot])
    plt.contour(alpha_grid, beta_grid, SOT_density.reshape(num, num), levels=20)
    if (name == 'SW'):
        plt.title("SW", fontsize=25)
    elif (name == 'MixSW'):
        plt.title("Mix-SW", fontsize=25)
    elif (name == 'SMixW'):
        plt.title("SMix-W", fontsize=25)
    plt.xlabel("$y^{(1)}$", fontsize=25)
    plt.ylabel("$y^{(2)}$", fontsize=25)
    plt.legend()
    plt.tight_layout()
    plt.show()
