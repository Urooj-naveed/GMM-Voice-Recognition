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

    # Return the GMM object with the initialized parameters
    return gmm

def compute_log_likelihood(X, means, covariances, weights):
    """
    Compute the log-likelihood of each data point under each Gaussian component.

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
        numpy.ndarray: A 2D numpy array of shape (X.shape[0], n_components)
            containing the log-likelihood of each data point under each Gaussian component.
    """

    # Get the number of data points and the number of features
    n_samples, n_features = X.shape

    # Get the number of Gaussian components
    n_components, _ = means.shape

    # Create an array to store the log-likelihood of each data point under each Gaussian component
    log_likelihoods = np.zeros((n_samples, n_components))

    # Compute the log-likelihood of each data point under each Gaussian component
    for i in range(n_components):
        # Get the covariance and precision matrices for the i-th Gaussian component
        covariance = covariances[i]
        covariance = (covariance + covariance.T) / 2.0  # make sure the covariance matrix is symmetric
        precision = np.linalg.inv(covariance)

        # Get the mean for the i-th Gaussian component
        mean = means[i]

        # Compute the log-likelihood of each data point under the i-th Gaussian component
        for j in range(n_samples):
            # Get the j-th data point
            x = X[j]

            # Compute the difference between the j-th data point and the mean of the i-th Gaussian component
            diff = x - mean

            # Compute the exponent of the multivariate normal distribution
            exponent = -0.5 * np.dot(diff.T, np.dot(precision, diff))

            # Compute the log-likelihood of the j-th data point under the i-th Gaussian component
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

    # Get the number of Gaussian components
    n_components = responsibilities.shape[1]

    # Compute the total responsibility of each Gaussian component
    total_resp = np.sum(responsibilities, axis=0)

    # Compute the new means of each Gaussian component
    means = np.dot(responsibilities.T, X) / total_resp[:, np.newaxis]

    # Compute the new covariances of each Gaussian component
    covariances = []
    for k in range(n_components):
        diff = X - means[k]
        cov_k = np.dot(responsibilities[:, k] * diff.T, diff) / total_resp[k]
        covariances.append(cov_k)

    # Compute the new weights of each Gaussian component
    weights = total_resp / X.shape[0]

    return means, covariances, weights


def fit(X, n_components, n_init=10, max_iter=100, tol=1e-6):
    """
    Fit a Gaussian Mixture Model to the data X using the Expectation-Maximization algorithm.

    Args:
        X (numpy.ndarray): A 2D numpy array where each row represents a feature vector.
        n_components (int): The number of Gaussian components in the mixture model.
        n_init (int): The number of times the initialization process is repeated.
        max_iter (int): The maximum number of iterations for the EM algorithm.
        tol (float): The convergence threshold for the EM algorithm.

    Returns:
        sklearn.mixture.GaussianMixture: A GaussianMixture object with the learned parameters.
    """
    from sklearn.mixture import GaussianMixture

    # Initialize the best log-likelihood to a very small value
    best_log_likelihood = -np.inf

    # Repeat the initialization process n_init times
    for i in range(n_init):

        # Initialize the parameters of the GMM
        print(n_components)
        gmm = GaussianMixture(n_components=n_components)
        gmm = initialize_parameters(X, gmm)

        # Iterate until convergence
        for iteration in range(max_iter):
            # Compute the log-likelihood of the current model
            log_likelihood = score(X, gmm.means_, gmm.covariances_, gmm.weights_) #gmm.score(X)

            nv_covariances = [np.linalg.inv(gmm.covariances_[k]) for k in range(gmm.n_components)]
            # Perform the E-step
            #_, responsibilities = expectation_step(X, gmm.means_, gmm.covariances_, gmm.weights_, gmm._get_inv_cov())
            _, responsibilities = expectation_step(X, gmm.means_, gmm.covariances_, gmm.weights_, nv_covariances)

            # Perform the M-step
            gmm.means_, gmm.covariances_, gmm.weights_ = maximization_step(X, responsibilities)

            # Check for convergence
            new_log_likelihood = score(X, gmm.means_, gmm.covariances_, gmm.weights_) #gmm.score(X)
            if new_log_likelihood - log_likelihood < tol:
                break
            else:
                log_likelihood = new_log_likelihood

        # If the current initialization yields a better model, update the best model
        if log_likelihood > best_log_likelihood:
            best_log_likelihood = log_likelihood
            best_gmm = gmm

    return best_gmm

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