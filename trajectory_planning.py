# Dependencies
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import dccp
import pdb
from tqdm import tqdm

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 15

# Parameters
dim = 2
NUM_ITERATIONS = 80
NUM_TRIALS = 1
ps = np.array([0.1, -0.1]).reshape(-1,1)
epsilons = np.array([0.1, 0.2]).reshape(-1,1)
A = np.array([[0.4, 1.5],
              [0., 0.9]])
B = np.array([[0.],
              [1.]])
T = 50
z_star = np.array([[0.],
                   [0.]])

x_low = -0.3
x_high = 0.3

# Variables for results
us = {}
us["SIP"] = []
us["Sample"] = []

for trial in tqdm(range(NUM_TRIALS)):
    ########################################################################################
    # Cutting-set method
    X = np.random.uniform(x_low, x_high, size=(dim, 15)) # Initial sample returns

    # Main loop
    for iter in range(NUM_ITERATIONS):
        ############################################
        # Solve problem (17)
        u = cp.Variable((T, 1))
        alpha = cp.Variable(ps.shape[0])
        beta  = cp.Variable(ps.shape[0])
        t = cp.Variable()

        constraints = []
        var = 0.
        for tau in range(T):
            var += np.linalg.matrix_power(A, T-1-tau)@B*u[tau, 0]
        for i in range(X.shape[1]):
            sample_x = X[:, i].reshape(-1,1)
            constraints += [cp.pnorm(np.linalg.matrix_power(A, T)@sample_x + var - z_star, 2) 
                            - beta.T@sample_x + beta.T@ps + alpha.T@epsilons - t <= 0]
        constraints += [alpha >= 0]
        constraints += [alpha+beta >= 0]
        constraints += [u>=-0.1, u<=0.1] 

        problem_1 = cp.Problem(objective=cp.Minimize(t), constraints=constraints)
        problem_1.solve()

        # Append result
        if iter == NUM_ITERATIONS - 1:
            us["SIP"] += [u.value.reshape(-1,1)]

        ############################################
        # Update x
        u_star = u.value.reshape(-1,1)
        beta_star = beta.value.reshape(-1,1)
        alpha_star = alpha.value.reshape(-1,1)
        t_star = t.value

        x = cp.Variable(dim)
        var = 0.
        for tau in range(T):
            var += np.linalg.matrix_power(A, T-1-tau)@B*u_star[tau, 0]
        var += -z_star
        var = var.reshape(-1,)
        obj = cp.Maximize(cp.pnorm(np.linalg.matrix_power(A, T)@x + var, 2) - beta_star.T@x)
        constraints = [x>=x_low, x<=x_high]
        problem_2 = cp.Problem(obj, constraints=constraints)
        if problem_1.value > 0:
            problem_2.solve(method='dccp')

            X = np.hstack([X, x.value.reshape(-1,1)])


    ########################################################################################
    # Sample-based Approach
    X_samples = np.random.uniform(x_low, x_high, size=(dim, 20)) # Initial sample returns

    for iter in range(NUM_ITERATIONS):

        ############################################
        # Find u
        u = cp.Variable((T,1))
        
        var = 0.
        for tau in range(T):
            var += np.linalg.matrix_power(A, T-1-tau)@B*u[tau, 0]
        terms = []
        for i in range(X_samples.shape[1]):
            sample_x = X_samples[:, i].reshape(-1,1)
            terms += [cp.pnorm(np.linalg.matrix_power(A, T)@sample_x + var - z_star, 2)]
        constraints = [u>=-0.1, u<=0.1] 
        
        problem_1 = cp.Problem(cp.Minimize(cp.sum(terms)), constraints=constraints)
        problem_1.solve()
        u_star = u.value.reshape(-1,1)

        if iter == NUM_ITERATIONS - 1:
            us["Sample"] += [u_star]

        ############################################
        # Update distribution
        n = X_samples.shape[1]
        X_vars = [cp.Variable((dim, 1)) for _ in range(n)]

        var = 0.
        for tau in range(T):
            var += np.linalg.matrix_power(A, T-1-tau)@B*u_star[tau, 0]
        terms = []
        for i in range(len(X_vars)):
            terms += [cp.pnorm(np.linalg.matrix_power(A, T)@X_vars[i] - var - z_star, 2)]

        constraints = [X_vars[_] >= x_low for _ in range(n)]
        constraints += [X_vars[_] <= x_high for _ in range(n)]
        for i in range(dim):
            constraints += [1/n*cp.sum([X_vars[_][i, 0] for _ in range(n)]) - ps[i] <= epsilons[i]]
            constraints += [1/n*cp.sum([X_vars[_][i, 0] for _ in range(n)]) - ps[i] >= -epsilons[i]]
        
        problem_2 = cp.Problem(cp.Maximize(cp.sum(terms)), constraints=constraints)
        problem_2.solve(method='dccp')
        X_samples = np.hstack([X_vars[_].value.reshape(-1,1) for _ in range(n)])
        
for key in us:
    us[key] = np.mean( np.hstack(us[key]), axis=1)

    var = 0.
    for tau in range(T):
        var += np.linalg.matrix_power(A, T-1-tau)@B*us[key][tau]

    print(key)
    print(var)
    print(us[key])
    print()


