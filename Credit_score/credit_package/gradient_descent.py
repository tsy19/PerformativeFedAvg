import numpy as np


def sigmoid(X):
    return 1 / (1 + np.exp(-X))


def grad(theta, X, y, lam, exp_tx=None):
    assert X.ndim == 2
    n = y.shape[0]
    if exp_tx is None:
        p = sigmoid(X @ theta)
    else:
        p = 1 / (1 + 1 / exp_tx)
    r = y - p
    g = - r.dot(X) / n
    g[:-1] += lam * theta[:-1]
    return g


def loss(theta, X, y, lam, exp_tx=None):
    # [Perdomo'21](https://arxiv.org/abs/2002.06673), Eq. (1)
    assert X.ndim == 2
    n = y.shape[0]
    xth = X @ theta
    l_log = - y.dot(xth) / n + np.mean(np.log(1 + np.exp(xth)))
    l_reg = lam / 2 * np.linalg.norm(theta[:-1])**2
    return l_log + l_reg


def get_constants(X, lam):
    n = X.shape[0]
    beta = np.linalg.norm(X, ord=2) ** 2 / (4.0 * n) + lam
    gamma = lam
    cond_num = beta / gamma
    return beta, gamma, cond_num


def gd(
    theta_init,
    grad_fn,
    loss_fn,
    stepsize,
    max_iter=1,
    tol=1e-8,
):
    theta = np.copy(theta_init)
    outputs = [[loss_fn(theta), np.copy(theta)]]
    for i in range(max_iter):
        grad = grad_fn(theta)
        theta -= stepsize * grad
        
        outputs.append([loss_fn(theta), np.copy(theta)])
        
        if np.linalg.norm(grad) < tol:
            #len(losses) >= 2 and np.abs(losses[-1] - losses[-2]) < tol:
            print('.', end='') #print(f'converged in {i}/{max_iter} steps')
            break
    else:
        print('x', end='') #print(f'does not converge in {max_iter} steps')
    return theta, outputs


def sgd(
    theta_init,
    Xs,
    ys,
    lam,
    stepsizes,
    max_iter=1,
    tol=1e-8,
):
    theta = np.copy(theta_init)
    loss_fn = lambda theta: loss(theta, Xs, ys, lam)
    outputs = [[loss_fn(theta), np.copy(theta)]]
    for i in range(max_iter):
        idx = np.random.choice(Xs.shape[0], size=1)
        Xi, yi = Xs[idx], ys[idx]
        theta -= stepsizes[i] * grad(theta, Xi, yi, lam)
        
        outputs.append([loss_fn(theta), np.copy(theta)])

    return theta, outputs
