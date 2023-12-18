from collections import defaultdict
import numpy as np
from tqdm.auto import trange
import mlflow
import wandb
from gradient_descent import *


def best_response(X, theta, eps, strat_features):
    """Get best response with linear utilities quadratic costs
    
    If X and eps are lists, return a list of best responses.
    
    Output:
        The same format as X.
    """
    if isinstance(eps, list): # X and eps are list
        return [best_response(_X, theta, _eps, strat_features)
                for _X, _eps in zip(X, eps)]
    X_strat = np.copy(X)
    X_strat[:, strat_features] -= eps * theta[strat_features]
    return X_strat


def perf_loss(theta, X, y, lam, eps, strat_features):
    Xs, ys = best_response(X, theta, eps, strat_features), y
    if isinstance(eps, list):
        Xs = np.concatenate(Xs)
        ys = np.concatenate(ys)
    return loss(theta, Xs, ys, lam)


def perf_gd(
    X_ls,
    y_ls,
    theta_init=None,
    lam=0,
    n_deployments=20,
    max_iter=100,
    eps_ls=None,
    strat_features=None,
    tol=1e-10,
):
    """
    Args:
        max_iter: max num of iterations for gradient descent.
            1: repeated gradient descent (RGD), or greedy deploy
            k: lazy deploy, with k gradient steps per deployment.
            Large k: repeated risk minimization (RRM)
    """
    if theta_init is None:
        theta = np.zeros(X_ls[0].shape[1])
    else:
        theta = np.copy(theta_init)

    N = len(y_ls) # n_client
    outputs = []
    for r in trange(n_deployments):
        Xs_ls, ys_ls = best_response(X_ls, theta, eps_ls, strat_features), y_ls
        Xs, ys = np.concatenate(Xs_ls), np.concatenate(ys_ls)

        beta, _, _ = get_constants(Xs, lam)
        stepsize = 1 / beta
        theta, gd_outputs = gd(
            theta,
            grad_fn=lambda theta: grad(theta, Xs, ys, lam),
            loss_fn=lambda theta: loss(theta, Xs, ys, lam),
            stepsize=stepsize,
            max_iter=max_iter,
            tol=tol,
        )

        outputs.extend(gd_outputs)
        
    return theta, outputs


def perf_gd_v1(
    data,
    theta_init=None,
    response_fn=None,
    get_grad_fn=None,
    get_loss_fn=None,
    get_stepsize=None,
    n_deployments=20,
    max_iter=100,
    tol=1e-10,
):
    """
    Args:
        max_iter: max num of iterations for gradient descent.
            1: repeated gradient descent (RGD), or greedy deploy
            k: lazy deploy, with k gradient steps per deployment.
            Large k: repeated risk minimization (RRM)
    """
    theta = np.copy(theta_init)
    outputs = []
    for r in trange(n_deployments):
        data = response_fn(data, theta)
        theta, gd_outputs = gd(theta,
            grad_fn=get_grad_fn(data),
            loss_fn=get_loss_fn(data),
            stepsize=get_stepsize(data),
            max_iter=max_iter,
            tol=tol,
        )
        outputs.extend(gd_outputs)        
    return theta, outputs


def perf_fed_avg(
    X_ls,
    y_ls,
    theta_init=None,
    lam=0,
    n_rounds=20,
    E=10, #max_iter=10,
    batch_size=None,
    p_ls=None,
    eps_ls=None,
    strat_features=None,
    full_participation=False,
    K=1,
    replace=True,
    # tol=1e-7,
    # callback=None,
    theta_PS=None,
    perf=True,
):
    """
    n_rounds: number of rounds of aggregation
    p_ls: aggregating weights for for each client. Inferred from X_ls if None.
    E: number of local gradient steps
    full_participation: full client participation in aggregation
    K: number of responded clients
    replace: Sampling K clients with replacement or without
    perf: whether to compute performative gradient or not (i.e., fed-avg)
    """
    N = len(X_ls) # number of total clients
    dim = X_ls[0].shape[1]
    theta_init = np.zeros(dim) if theta_init is None else np.copy(theta_init)

    if full_participation:
        K, replace = N, False

    if p_ls is None:
        p_ls = np.array([X.shape[0] for X in X_ls])
        p_ls = p_ls / p_ls.sum()

    outputs = []
    global_theta = np.copy(theta_init)
    for r in trange(n_rounds):
        subset = np.arange(N) if full_participation else np.random.choice(N, size=K, replace=replace)
        local_theta = {}
        for k in subset:
            theta = np.copy(global_theta)
            X, y, eps = X_ls[k], y_ls[k], eps_ls[k]
            for t in range(r * E, (r+1) * E):
                # Strategic response
                Xs, ys = best_response(X, theta, 0, strat_features), y

                # Stochastic batch
                if batch_size is not None:
                    idx = np.random.choice(Xs.shape[0], size=batch_size, replace=False)
                else:
                    idx = slice(None)
                Xi, yi = Xs[idx], ys[idx]

                # Gradient descent step
                stepsize = 50 / (t + 10**4)
                # stepsize = 1 / get_constants(Xs, lam)[0] # not converging
                gradient = grad(theta, Xi, yi, lam)
                theta -= stepsize * gradient

            local_theta[k] = theta
        global_theta = N / K * sum([p_ls[k] * local_theta[k]
                                    for k in subset])
        if r % 100 == 0:
            deviations = [np.linalg.norm(local_theta[k] - global_theta) ** 2
                          for k in subset]
            concensus_error = deviations @ p_ls[subset]

            loss_val = perf_loss(global_theta, X_ls, y_ls, lam, eps_ls, strat_features)
            theta_dist = np.linalg.norm(global_theta - theta_PS)

            mlflow.log_metrics({
                'loss': loss_val,
                'theta_dist': theta_dist,
                'concensus_error': concensus_error,
                # 'test_accuracy':
            },
                step=r,
            )
    return global_theta, outputs
