import numpy as np
import sys
import warnings
from scipy.special import logsumexp
from sklearn.utils import check_random_state
from sklearn.exceptions import ConvergenceWarning



def fit(model, X):
    """
    Train a Gaussian mixture model using the expectation-maximization algorithm.

    Args:
        model (sklearn.mixture.GaussianMixture): An instance of the GaussianMixture class from the sklearn module.
        X (numpy.ndarray): A 2D numpy array where each row represents a feature vector.

    Returns:
        model
    """
    model._validate_params()
    X = model._validate_data(X, dtype=[np.float64, np.float32], ensure_min_samples=2)
    if X.shape[0] < model.n_components:
        raise ValueError(
            "Expected n_samples >= n_components "
            f"but got n_components = {model.n_components}, "
            f"n_samples = {X.shape[0]}"
        )
    model._check_parameters(X)

    # if we enable warm_start, we will have a unique initialisation
    do_init = not (model.warm_start and hasattr(model, "converged_"))
    n_init = model.n_init if do_init else 1

    max_lower_bound = -np.inf
    model.converged_ = False

    random_state = check_random_state(model.random_state)

    n_samples, _ = X.shape
    for init in range(n_init):
        model._print_verbose_msg_init_beg(init)

        if do_init:
            model._initialize_parameters(X, random_state)

        lower_bound = -np.inf if do_init else model.lower_bound_

        if model.max_iter == 0:
            best_params = model._get_parameters()
            best_n_iter = 0
        else:
            for n_iter in range(1, model.max_iter + 1):
                prev_lower_bound = lower_bound

                log_prob_norm, log_resp = model._e_step(X)
                model._m_step(X, log_resp)
                lower_bound = model._compute_lower_bound(log_resp, log_prob_norm)

                change = lower_bound - prev_lower_bound
                model._print_verbose_msg_iter_end(n_iter, change)

                if abs(change) < model.tol:
                    model.converged_ = True
                    break

            model._print_verbose_msg_init_end(lower_bound)

            if lower_bound > max_lower_bound or max_lower_bound == -np.inf:
                max_lower_bound = lower_bound
                best_params = model._get_parameters()
                best_n_iter = n_iter

    # Should only warn about convergence if max_iter > 0, otherwise
    # the user is assumed to have used 0-iters initialization
    # to get the initial means.
    if not model.converged_ and model.max_iter > 0:
        warnings.warn(
            "Initialization %d did not converge. "
            "Try different init parameters, "
            "or increase max_iter, tol "
            "or check for degenerate data." % (init + 1),
            ConvergenceWarning,
        )

    model._set_parameters(best_params)
    model.n_iter_ = best_n_iter
    model.lower_bound_ = max_lower_bound

        # Always do a final e-step to guarantee that the labels returned by
        # fit_predict(X) are always consistent with fit(X).predict(X)
        # for any value of max_iter and tol (and any random_state).
    _, log_resp = model._e_step(X)
    return model


