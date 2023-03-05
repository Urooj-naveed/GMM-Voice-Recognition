import numpy as np
from scipy.special import logsumexp

def initialize_parameters(X, gmm):
    """
    Initialize the means, covariances, and weights of the GMM from an object of class GaussianMixture
    and the feature matrix X.

    Args:
        gmm (sklearn.mixture.GaussianMixture): An object of the class GaussianMixture.
        X (numpy.ndarray): A 2D numpy array where each row represents a feature vector.

    Returns:
        sklearn.mixture.GaussianMixture: A GaussianMixture object with the initialized parameters.
    """
    # Initialize means randomly
    means = np.random.randn(gmm.n_components, X.shape[1])
    
    # Initialize covariances as identity matrices
    covariances = [np.eye(X.shape[1]) for i in range(gmm.n_components)]
    
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
    """
    Perform the E-step of the EM algorithm.

    Args:
        X (numpy.ndarray): A 2D numpy array where each row represents a feature vector.
        means (numpy.ndarray): A 2D numpy array of shape (n_components, X.shape[1])
            containing the means of the Gaussian components.
        covariances (list of numpy.ndarrays): A list of length n_components,
            where each element is a 2D numpy array of shape (X.shape[1], X.shape[1])
            containing the covariance matrix of the Gaussian components.
        weights (numpy.ndarray): A 1D numpy array of shape (n_components)
            containing the weights of the Gaussian components.
        inv_covariances (list of numpy.ndarrays): A list of length n_components,
            where each element is a 2D numpy array of shape (X.shape[1], X.shape[1])
            containing the inverse of the covariance matrix of the Gaussian components.

    Returns:
        Tuple of numpy arrays: (log_likelihoods, responsibilities), where
            log_likelihoods (numpy.ndarray): A 1D numpy array of shape (X.shape[0]) containing
                the log-likelihood of each data point under the GMM.
            responsibilities (numpy.ndarray): A 2D numpy array of shape (X.shape[0], n_components)
                containing the responsibility of each data point for each Gaussian component.
    """
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

        # Update the responsibilities matrix
        responsibilities[:, k] = np.log(weights[k]) - 0.5 * np.log(np.linalg.det(covariances[k])) + exponent

    # Use logsumexp to compute the log-likelihood of each data point under the entire GMM
    log_likelihoods = logsumexp(responsibilities, axis=1)
    responsibilities = np.exp(responsibilities - log_likelihoods[:, np.newaxis])

    return log_likelihoods, responsibilities


def maximization_step(X, responsibilities):
    """
    Perform the M-step of the EM algorithm.

    Args:
        X (numpy.ndarray): A 2D numpy array where each row represents a feature vector.
        responsibilities (numpy.ndarray): A 2D numpy array of shape (X.shape[0], n_components)
            containing the responsibility of each data point for each Gaussian component.

    Returns:
        Tuple of numpy arrays: (means, covariances, weights), where
            means (numpy.ndarray): A 2D numpy array of shape (n_components, X.shape[1])
                containing the means of the Gaussian components.
            covariances (list of numpy.ndarrays): A list of length n_components,
                where each element is a 2D numpy array of shape (X.shape[1], X.shape[1])
                containing the covariance matrix of the Gaussian components.
            weights (numpy.ndarray): A 1D numpy array of shape (n_components)
                containing the weights of the Gaussian components.
    """
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
    weights = sum_resp / n_samples

    return means, covariances, weights

def fit(gmm, X, max_iter=100, tol=1e-4):
    # Initialize the parameters of the GMM
    gmm = initialize_parameters(X, gmm)
    
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
    """
    Compute the per-sample average log-likelihood of the given data X, where X is a 2D numpy array
    where each row represents a feature vector.

    Args:
        X (numpy.ndarray): A 2D numpy array where each row represents a feature vector.
        means (numpy.ndarray): A 2D numpy array of shape (n_components, X.shape[1])
            containing the means of the Gaussian components.
        covariances (list of numpy.ndarrays): A list of length n_components,
            where each element is a 2D numpy array of shape (X.shape[1], X.shape[1])
            containing the covariance matrix of the Gaussian components.
        weights (numpy.ndarray): A 1D numpy array of shape (n_components)
            containing the weights of the Gaussian components.

    Returns:
        float: The per-sample average log-likelihood of the given data X.
    """
    # Compute the log-likelihoods of the data
    log_likelihoods = compute_log_likelihood(X, means, covariances, weights)

    # Compute the per-sample average log-likelihood
    return np.mean(log_likelihoods)