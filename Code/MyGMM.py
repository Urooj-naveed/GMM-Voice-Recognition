import os
import pickle
import warnings
import numpy as np

warnings.filterwarnings("ignore")


class MyGMM:
    """
    This class implements the Gaussian mixture model algorithm from scratch.
    """

    def __init__(self, n_components=1, tol=1e-4, max_iter=200, covariance_type='diag', n_init=1):
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
        elif self.covariance_type == 'full':
            self.covariances_ = np.array([np.cov(X.T) for i in range(self.n_components)])

    def pdf(self, X, mean=None, cov=None):
     if mean is None:
        mean = self.means_
     if cov is None:
        cov = self.covariances_
     X = X[:, np.newaxis, :]  # add extra dimension to match shape of mean
     mean = mean[np.newaxis, :, :]
     det = np.linalg.det(cov)
     inv = np.linalg.inv(cov)
     norm_const = 1.0 / (np.power((2*np.pi), float(X.shape[2])/2) * np.power(det,1.0/2))
     X = X[:, np.newaxis, :]  # add extra dimension to match shape of mean
     mean = mean[np.newaxis, :, :]  # add extra dimension to mean
     x_mu = X - mean.swapaxes(0, 1)
     result = np.power(np.e, -0.5 * np.sum(np.multiply(np.matmul(x_mu,inv), x_mu), axis=2)) * norm_const
     return result

    def fit(self, X):
        self.initialize(X)
        self.weights_old_ = np.zeros_like(self.weights_)

        # EM algorithm
        for i in range(self.max_iter):
            # Expectation step
            resp = self.pdf(X) * self.weights_[:, np.newaxis]
            self.resp_ = resp / resp.sum(axis=1)[:, np.newaxis]

            # Maximization step
            self.weights_ = self.resp_.sum(axis=0)
            self.means_ = (self.resp_[:,:,np.newaxis] * X[:,np.newaxis,:]).sum(axis=0) / self.resp_.sum(axis=0)[:,np.newaxis]
            for j in range(self.n_components):
                x_mu = X - self.means_[j]
                self.covariances_[j] = np.einsum('ijk,ijl->kl', x_mu, x_mu*self.resp_[:,j,np.newaxis], optimize=True) / self.resp_[:,j].sum()

            # Check for convergence
            if np.abs(self.weights_ - self.weights_old_).max() < self.tol:
                break
            self.weights_old_ = self.weights_.copy()