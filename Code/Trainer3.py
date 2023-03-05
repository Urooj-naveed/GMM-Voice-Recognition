import numpy as np
from scipy.special import logsumexp

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


def expectation_step(X, means, covariances, weights, inv_covariances):
    # Get the number of Gaussian components
    n_components = len(weights)

    # Initialize the responsibilities matrix with zeros
    responsibilities = np.zeros((X.shape[0], n_components))

    # Compute the log-likelihood of each data point under each Gaussian component
    for k in range(n_components):
        # Compute the difference between each data point and the mean of the kth Gaussian component
        diff = X - means[k]

        # Compute the exponential term in the multivariate normal distribution
        exponent = -0.5 * np.sum(np.dot(diff, inv_covariances[k]) * diff, axis=1)
        #exponent = -0.5 * np.einsum('ij,ijk,ik->i', diff, inv_covariances, diff)

        # Update the responsibilities matrix
        responsibilities[:, k] = np.log(weights[k]) - 0.5 * np.log(np.linalg.det(covariances[k])) + exponent

    # Use logsumexp to compute the log-likelihood of each data point under the entire GMM
    log_likelihoods = logsumexp(responsibilities, axis=1)
    responsibilities = np.exp(responsibilities - log_likelihoods[:, np.newaxis])

    return log_likelihoods, responsibilities


def maximization_step(X, responsibilities):
    # Get the number of Gaussian components and the total number of data points
    n_components = responsibilities.shape[1]
    n_samples = X.shape[0]

    # Compute the sum of responsibilities for each component
    sum_resp = np.sum(responsibilities, axis=0)

    # Update the means of the Gaussian components
    means = np.dot(responsibilities.T, X) / sum_resp[:, np.newaxis]

    # Update the covariances of the Gaussian components
    covariances = []
    for k in range(n_components):
        diff = X - means[k]
        cov_k = np.dot(responsibilities[:, k] * diff.T, diff) / sum_resp[k]
        covariances.append(cov_k)

    # Update the weights of the Gaussian components
    #weights = sum_resp / n_samples
    weights = np.mean(responsibilities, axis=0)

    return means, covariances, weights

def fit(gmm, X, max_iter=100, tol=1e-4, random_state=None):
    # Initialize the parameters of the GMM
    gmm = initialize_parameters(X, gmm, random_state=random_state)
    
    prev_log_likelihood = None
    inv_covariances = [np.linalg.inv(gmm.covariances_[k]) for k in range(gmm.n_components)]

    for i in range(max_iter):
        # Perform the E-step
        log_likelihoods, responsibilities = expectation_step(X, gmm.means_, gmm.covariances_, gmm.weights_, inv_covariances)
        
        # Compute the log-likelihood of the data under the GMM
        log_likelihood = np.sum(log_likelihoods)
        
        # Check for convergence
        if prev_log_likelihood is not None and np.abs(log_likelihood - prev_log_likelihood) < tol:
            break
        
        prev_log_likelihood = log_likelihood
        
        # Perform the M-step
        gmm.means_, gmm.covariances_, gmm.weights_ = maximization_step(X, responsibilities)
    
    return gmm

def score(X, means, covariances, weights):
    # Compute the log-likelihoods of the data
    log_likelihoods = compute_log_likelihood(X, means, covariances, weights)

    # Compute the per-sample average log-likelihood
    return np.mean(log_likelihoods)