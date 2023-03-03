import os
import pickle
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


class MyGMM:
    """
    This class implements the Gaussian mixture model algorithm from scratch.
    """

    def __init__(self, n_components=2, tol=1e-4, max_iter=200, covariance_type='diag', n_init=1):
        self.n_components = n_components
        self.tol = tol
        self.max_iter = max_iter
        self.covariance_type = covariance_type
        self.n_init = n_init

    def initialize(self, X):
    # Initialize the weights, means, and covariances
     self.weights_ = np.full(shape=self.n_components, fill_value=1/self.n_components)
     self.means_ = X[np.random.choice(range(X.shape[0]), self.n_components, replace=False)]
     self.covariances_ = np.zeros((self.n_components, X.shape[1], X.shape[1]))
     if self.covariance_type == 'diag':
        for i in range(self.n_components):
            self.covariances_[i] = np.diag(np.random.rand(X.shape[1]))
            #print(self.covariances_[i])
     elif self.covariance_type == 'full':
        cov = np.cov(X.T)
        self.covariances_ = np.tile(cov, (self.n_components, 1, 1))  # reshape to 3D matrix
        #print(self.covariances_)
    
    def log_likelihood(self, X, weights, means, covariances):
        n_components = len(weights)
        log_det_covariances = np.zeros(n_components)
        precision_matrices = np.zeros((n_components, X.shape[1], X.shape[1]))
        log_likelihoods = np.zeros((X.shape[0], n_components))
        for i in range(n_components):
            sign, logdet = np.linalg.slogdet(covariances[i])
            log_det_covariances[i] = logdet
            precision_matrices[i] = np.linalg.inv(covariances[i])
            log_likelihoods[:, i] = -0.5 * (np.sum((X - means[i]) @ precision_matrices[i] * (X - means[i]), axis=1)
                                            + X.shape[1] * np.log(2*np.pi) + log_det_covariances[i])
        return log_likelihoods

    def fit(self, X):
        self.initialize(X)
        self.weights_old_ = np.zeros_like(self.weights_)

        # EM algorithm
        for i in range(self.max_iter):
            # Expectation step
            resp = self.log_likelihood(X, self.weights_, self.means_, self.covariances_) + np.log(self.weights_[np.newaxis, :])
            self.resp_ = np.exp(resp - np.max(resp, axis=1, keepdims=True))
            

            # Maximization step
            self.weights_ = self.resp_.sum(axis=0)
            self.means_ = (self.resp_[:,:,np.newaxis] * X[:,np.newaxis,:]).sum(axis=0) / self.resp_.sum(axis=0)[:,np.newaxis]
            for j in range(self.n_components):
                x_mu = X - self.means_[j]
                self.covariances_[j] = np.einsum('ik,il,j->kl', x_mu, x_mu*self.resp_[:,j,np.newaxis], self.resp_[:,j]) / self.resp_[:,j].sum()
            # Check for convergence
            if np.abs(self.weights_ - self.weights_old_).max() < self.tol:
                break
            self.weights_old_ = self.weights_.copy()
    def score(self, X):
        # Compute the log-likelihood of each sample under each component
        log_likelihoods = self.log_likelihood(X, self.weights_, self.means_, self.covariances_)

        # Compute the per-sample log-likelihood, which is the log-sum-exp of the log-likelihoods
        log_likelihoods_samples = np.log(np.sum(np.exp(log_likelihoods), axis=1))

        # Compute the average log-likelihood over all samples
        avg_log_likelihood = np.mean(log_likelihoods_samples)

        # Return the negative of the average log-likelihood, since higher values are better
        return abs(-avg_log_likelihood)
    
np.random.seed(42)
df = pd.read_excel("data_points.xlsx")
X_train = df.iloc[0:200, 0:14].to_numpy()
X_test = df.iloc[200:400, 0:14].to_numpy()
gmm = MyGMM(n_components=2, tol=1e-4, max_iter=200, covariance_type='diag', n_init=1)
gmm.fit(X_train)
accuracy = gmm.score(X_test)
print("Accuracy:", accuracy)
