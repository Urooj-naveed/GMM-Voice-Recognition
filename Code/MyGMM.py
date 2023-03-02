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
        loglik = np.zeros((X.shape[0], n_components))
        for i in range(n_components):
            norm_const = -0.5 * (np.log(2*np.pi) + np.log(np.linalg.det(covariances[i])))
            precision_matrix = np.linalg.inv(covariances[i])
            loglik[:, i] = norm_const - 0.5 * np.sum((X - means[i]) @ precision_matrix * (X - means[i]), axis=1)
        return loglik

    def fit(self, X):
        self.initialize(X)
        self.weights_old_ = np.zeros_like(self.weights_)

        # EM algorithm
        for i in range(self.max_iter):
            # Expectation step
            resp = self.log_likelihood(X, self.weights_, self.means_, self.covariances_) * self.weights_[np.newaxis, :]
            self.resp_ = resp / resp.sum(axis=1)[:, np.newaxis]

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

np.random.seed(42)
df = pd.read_excel("data_points.xlsx")
Y = df.to_numpy()
#print(df.rows)
#print('Size of the DataFrame:', df.shape)
#print('Data type of each column:\n', df.dtypes)
#print(Y)
gmm = MyGMM(n_components=2, tol=1e-4, max_iter=200, covariance_type='diag', n_init=1)
gmm.fit(Y)
