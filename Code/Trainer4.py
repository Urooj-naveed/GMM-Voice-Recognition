import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

def initialize_parameters(X, gmm, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    
    # Initialize means randomly
    means = np.random.randn(gmm.n_components, X.shape[1])
    
    # Initialize covariances as identity matrices
    covariances = np.tile(np.eye(X.shape[1])[np.newaxis], [gmm.n_components, 1, 1])
    
    # Initialize weights as uniform probabilities
    weights = np.ones(gmm.n_components) / gmm.n_components

    # Update the parameters of the GMM
    gmm.means_ = means
    gmm.covariances_ = covariances
    gmm.weights_ = weights
    #gmm.fit(X)

    return gmm

def compute_log_likelihood(X, means, covariances, weights):
    n_samples, n_features = X.shape
    n_components, _ = means.shape

    log_likelihoods = np.zeros((n_samples, n_components))

    for i in range(n_components):
        covariance = covariances[i]
        covariance = (covariance + covariance.T) / 2.0  # make sure the covariance matrix is symmetric
        precision = np.linalg.inv(covariance)
        mean = means[i]
        for j in range(n_samples):
            x = X[j]
            diff = x - mean
            exponent = -0.5 * np.dot(diff.T, np.dot(precision, diff))
            log_likelihoods[j, i] = np.log(weights[i]) + exponent - 0.5 * np.log(np.linalg.det(covariance))

    return log_likelihoods


def expectation_step(X, means, L, weights):
    log_likelihoods = np.zeros((X.shape[0], weights.shape[0]))
    responsibilities = np.zeros((X.shape[0], weights.shape[0]))

    # Compute the log-likelihood of each data point under each mixture component
    for k in range(weights.shape[0]):
        log_likelihoods[:, k] = multivariate_normal.logpdf(X, mean=means[k], cov=np.dot(L[k], L[k].T)) + np.log(weights[k])

    # Compute the responsibilities
    responsibilities = np.exp(log_likelihoods - logsumexp(log_likelihoods, axis=1)[:, np.newaxis])

    return log_likelihoods, responsibilities


def maximization_step(X, responsibilities, L):
    means = np.zeros((responsibilities.shape[1], X.shape[1]))
    covariances = np.zeros((responsibilities.shape[1], X.shape[1], X.shape[1]))
    weights = np.zeros(responsibilities.shape[1])

    # Update the means
    for k in range(responsibilities.shape[1]):
        means[k] = np.sum(responsibilities[:, k][:, np.newaxis] * X, axis=0) / np.sum(responsibilities[:, k])

    # Update the covariances
    for k in range(responsibilities.shape[1]):
        #print(responsibilities[:, k][:, np.newaxis].shape)
        #print((X - means[k]).shape)
        #covariances[k] = (1 / np.sum(responsibilities[:, k])) * np.dot((responsibilities[:, k][:, np.newaxis] * (X - means[k])).T, (L[k].T))
        covariances[k] = (1 / np.sum(responsibilities[:, k])) * np.dot((responsibilities[:, k][:, np.newaxis] * (X - means[k])).T, (responsibilities[:, k][:, np.newaxis] * (X - means[k])))


    # Update the weights
    weights = np.sum(responsibilities, axis=0) / X.shape[0]

    return means, covariances, weights

def fit(gmm, X, max_iter=100, tol=1e-4, random_state=None):
    # Initialize the parameters of the GMM
    gmm = initialize_parameters(X, gmm, random_state=random_state)

    prev_log_likelihood = None
    L = np.zeros((gmm.n_components, X.shape[1], X.shape[1]))
    for k in range(gmm.n_components):
        L[k] = np.linalg.cholesky(gmm.covariances_[k])

    for i in range(max_iter):
        # Perform the E-step
        log_likelihoods, responsibilities = expectation_step(X, gmm.means_, L, gmm.weights_)

        # Compute the log-likelihood of the data under the GMM
        log_likelihood = np.sum(log_likelihoods)

        # Check for convergence
        if prev_log_likelihood is not None and np.abs(log_likelihood - prev_log_likelihood) < tol:
            break

        prev_log_likelihood = log_likelihood

        # Perform the M-step
        gmm.means_, gmm.covariances_, gmm.weights_ = maximization_step(X, responsibilities, L)

    return gmm

def score(X, means, covariances, weights):
    # Compute the log-likelihoods of the data
    log_likelihoods = compute_log_likelihood(X, means, covariances, weights)

    # Compute the per-sample average log-likelihood
    return np.mean(log_likelihoods)