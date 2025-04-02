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
dim = 3
NUM_ITERATIONS = 50
NUM_TRIALS = 2

VOL = True # indicates whether volatility constraints are present
if VOL:
    ps = np.array([0.3, -0.1, 0.2]).reshape(-1,1)
    epsilons = np.array([0.4, 0.2, 0.1]).reshape(-1,1)

    ps_vol = np.array([0.34, 0.26, 0.1]).reshape(-1,1)
    epsilons_vol = np.array([0.05, 0.05, 0.05]).reshape(-1,1)

# Variables for results
us = {}
us["SIP"] = []
us["Sample"] = []

for trial in tqdm(range(NUM_TRIALS)):
    ########################################################################################
    # Cutting-set method
    X = np.random.uniform(-1., 1., size=(dim, 150)) # Initial sample returns

    # Main loop
    for iter in range(NUM_ITERATIONS):
        ############################################
        # Solve problem (17)
        u = cp.Variable(dim)
        alpha = cp.Variable(ps.shape[0])
        beta  = cp.Variable(ps.shape[0])
        if VOL:
            alpha_vol = cp.Variable(ps_vol.shape[0])
            beta_vol  = cp.Variable(ps_vol.shape[0])
        t = cp.Variable()

        if not VOL:
            constraints = []
            for i in range(X.shape[1]):
                sample_x = X[:, i].reshape(-1,1)
                constraints += [-u.T@sample_x - beta.T@sample_x + beta.T@ps + alpha.T@epsilons - t <= 0]
            constraints += [alpha >= 0]
            constraints += [alpha+beta >= 0]
        if VOL:
            constraints = []
            for i in range(X.shape[1]):
                sample_x = X[:, i].reshape(-1,1)
                constraints += [-u.T@sample_x - beta.T@sample_x + beta.T@ps + alpha.T@epsilons
                                              - beta_vol.T@sample_x**2 + beta_vol.T@ps_vol + alpha_vol.T@epsilons_vol - t <= 0]
            constraints += [alpha >= 0]
            constraints += [alpha+beta >= 0]
            constraints += [alpha_vol >= 0]
            constraints += [alpha_vol+beta_vol >= 0]
        constraints += [cp.sum(u)==1, u>=0] #u>=-2, u<=2] 

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
        if VOL:
            alpha_vol_star = alpha_vol.value.reshape(-1,1)
            beta_vol_star  = beta_vol.value.reshape(-1,)
        t_star = t.value

        x = cp.Variable(dim)
        if not VOL:
            obj = cp.Maximize(-u_star.T@x - beta_star.T@x)
        if VOL:
            obj = cp.Maximize(-u_star.T@x - beta_star.T@x -  cp.sum(cp.multiply( beta_vol_star, cp.square(x)))     )
        constraints = [x>=-1, x<=1]
        problem_2 = cp.Problem(obj, constraints=constraints)
        if problem_1.value >= 0: # if algorithm not converged
            problem_2.solve()

            X = np.hstack([X, x.value.reshape(-1,1)])


    ########################################################################################
    # Sample-based Approach
    X_samples = np.random.uniform(-1., 1., size=(dim, 100)) # Initial sample returns

    for iter in range(NUM_ITERATIONS):

        ############################################
        # Find u
        u = cp.Variable(dim)
        
        terms = []
        for i in range(X_samples.shape[1]):
            sample_x = X_samples[:, i].reshape(-1,1)
            terms += [-u.T@sample_x]
        constraints = [cp.sum(u)==1, u>=0] #u>=-2, u<=2] 
        
        problem_1 = cp.Problem(cp.Minimize(cp.sum(terms)), constraints=constraints)
        problem_1.solve()
        u_star = u.value.reshape(-1,1)

        if iter == NUM_ITERATIONS - 1:
            us["Sample"] += [u_star]

        ############################################
        # Update distribution
        n = X_samples.shape[1]
        X_vars = [cp.Variable((dim, 1)) for _ in range(n)]

        terms = []
        for i in range(len(X_vars)):
            terms += [-u_star.T@X_vars[i]]

        constraints = [X_vars[_] >= -1 for _ in range(n)]
        constraints += [X_vars[_] <= 1 for _ in range(n)]
        for i in range(dim):
            constraints += [1/n*cp.sum([X_vars[_][i, 0] for _ in range(n)]) - ps[i] <= epsilons[i]]
            constraints += [1/n*cp.sum([X_vars[_][i, 0] for _ in range(n)]) - ps[i] >= -epsilons[i]]
            if VOL:
                constraints += [1/n*cp.sum([X_vars[_][i, 0]**2 for _ in range(n)]) - ps_vol[i] <= epsilons_vol[i]]
                constraints += [1/n*cp.sum([X_vars[_][i, 0]**2 for _ in range(n)]) - ps_vol[i] >= -epsilons_vol[i]]
        
        problem_2 = cp.Problem(cp.Maximize(cp.sum(terms)), constraints=constraints)
        if VOL:
            problem_2.solve(method='dccp')
        else:
            problem_2.solve()
        X_samples = np.hstack([X_vars[_].value.reshape(-1,1) for _ in range(n)])
        

for key in us:
    us[key] = np.mean( np.hstack(us[key]), axis=1)

width =0.3
plt.bar(1+np.arange(dim), us["Sample"], width=width, label="best-response")
plt.bar(1+np.arange(dim)+width, us["SIP"], width=width, label="cutting-set")
plt.xlabel("Asset")
plt.ylabel("Portfolio Weight")
plt.xticks([1,2,3])
plt.legend()
plt.show()
