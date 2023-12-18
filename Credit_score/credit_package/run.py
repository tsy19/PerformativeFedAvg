import argparse
from sklearn.linear_model import LogisticRegression
from data_prep import load_data
from gradient_descent import *
from performative import *
from utils import *
import mlflow


training_data_file = 'cs-training.csv'
X, Y, df_data = load_data(training_data_file)
# Test data has no labels
# test_data_file = 'cs-test.csv'
# X_test, Y_test, df_data_test = load_data(test_data_file)



# ERM solution
theta_init = np.zeros(X.shape[1])
lam = 10 / X.shape[0]
reg = LogisticRegression(penalty='l2', fit_intercept=True, C=1/lam/X.shape[0])
reg.fit(X[:,:-1], Y)
theta_ERM = np.append(reg.coef_[0], reg.intercept_)


# Performative prediction
strat_features = np.array([1, 6, 8]) - 1
n_clients = 10
M = X.shape[0] // n_clients


# arguments
parser = argparse.ArgumentParser()
parser.add_argument('-K', '--K', type=int, default=10)
parser.add_argument('-E', '--E', type=int, default=5)
parser.add_argument('--eps_max', type=float, default=11)
parser.add_argument('--eps_min', type=float, default=9)
parser.add_argument('-r', '--replace', type=bool, default=False)
parser.add_argument('-s', '--seed', type=int, default=None,
                    help='Seed for SGD, not for eps_ls generation.')
parser.add_argument('-b', '--batch_size', type=int, default=1)
parser.add_argument('-n', '--n_rounds', type=int, default=int(1e5))
parser.add_argument('-p', '--perf', type=bool, default=True)
parser.add_argument('-d', '--data_type', type=str, default='5-partition')
args = parser.parse_args()

data_type = args.data_type
batch_size = args.batch_size if args.batch_size > 0 else None
np.random.seed(0)
unif = np.random.rand(n_clients)
eps_range = args.eps_max - args.eps_min
eps_ls = (args.eps_min + unif * eps_range).tolist()
replace = args.replace
perf = args.perf
full_participation = args.K == n_clients

np.random.seed(args.seed)

# data
if data_type == 'static': # identical static data
    # X_ls = [np.copy(X[:M]) for _ in range(n_clients)]
    # y_ls = [np.copy(Y[:M]) for _ in range(n_clients)]
    X_ls = [np.copy(X) for _ in range(n_clients)]
    y_ls = [np.copy(Y) for _ in range(n_clients)]
elif data_type == '5-partition':
    X_ls, y_ls = [], []
    for k in range(n_clients):
        s = slice(k * M, (k+1) * M)
        X_ls.append(X[s])
        y_ls.append(Y[s])

with mlflow.start_run():
    # PS solution
    theta_PS, outputs_PS = perf_gd(
        X_ls,
        y_ls,
        theta_init=theta_ERM,
        lam=lam,
        n_deployments=200,
        max_iter=50,
        eps_ls=eps_ls,
        strat_features=strat_features,
        tol=1e-12,
    )
    fig1, fig2 = plot_gd(outputs_PS)
    mlflow.log_figure(fig1, 'loss.png')
    mlflow.log_figure(fig2, 'dtheta.png')


    # PS fedavg
    mlflow.log_params({
        'data': data_type,
        'lam': lam,
        'E': args.E,
        'eps': eps_ls,
        'N': n_clients,
        'K': args.K,
        'batch_size': batch_size,
        'replace': replace,
        'seed': args.seed,
        'n_rounds': args.n_rounds,
    })
    theta_PF, outputs_PF = perf_fed_avg(
        X_ls,
        y_ls,
        theta_init=np.copy(theta_ERM), #np.zeros(theta_ERM.shape)
        lam=lam,
        n_rounds=args.n_rounds,
        E=args.E,
        batch_size=batch_size,
        p_ls=None, # inferred from data
        eps_ls=eps_ls,
        strat_features=strat_features,
        full_participation=full_participation,
        K=args.K,
        replace=replace,
        theta_PS=theta_PS,
        perf=perf,
    )
