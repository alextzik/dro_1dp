# Dependencies
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import pdb

# Parameters
gamma = 1e-3
Sigma = np.eye(10)
NUM_ITERATIONS = 3
ps = np.array([0.3, -0.1]).reshape(-1,1)
epsilons = np.array([0.4, 0.2]).reshape(-1,1)

obj_vals = {}
us = {}


############################################
# Cutting set method
X = np.random.uniform(-1., 1., size=(10, 10)) # Initial sample returns

obj_vals["SIP"] = []
us["SIP"] = []
# Main loop
for iter in range(NUM_ITERATIONS):

    ############################################
    # Solve approximate version of problem (13)
    u = cp.Variable(10)
    alpha = cp.Variable(2)
    beta  = cp.Variable(2)
    t = cp.Variable()

    constraints = []
    for i in range(X.shape[1]):
        sample_x = X[:, i].reshape(-1,1)
        constraints += [-u.T@sample_x + gamma*cp.quad_form(u, Sigma) - beta.T@sample_x[0:2, :] + beta.T@ps + alpha.T@epsilons - t<= 0]
    constraints += [alpha>=0]
    constraints += [alpha+beta >=0]
    constraints += [cp.sum(u)==1, u>=-2, u<=2]

    problem_1 = cp.Problem(objective=cp.Minimize(t), constraints=constraints)
    problem_1.solve()
    obj_vals["SIP"] += [problem_1.value]
    us["SIP"] += [u.value.reshape(-1,1)]


    ############################################
    # Update x
    u_star = u.value.reshape(-1,1)
    beta_star = beta.value.reshape(-1,1)
    alpha_star = alpha.value.reshape(-1,1)
    t_star = t.value

    x = cp.Variable(10)
    obj = cp.Maximize(-u_star.T@x - beta_star.T@x[0:2])
    constraints = [x>=-1, x<=1]
    problem_2 = cp.Problem(obj, constraints=constraints)
    problem_2.solve()
    print(problem_2.value)

    X = np.hstack([X, x.value.reshape(-1,1)])


############################################
# Sample-based Approach
obj_vals["Sample"] = []
us["Sample"] = []

X = np.random.uniform(-1., 1., size=(10, 100)) # Initial sample returns

for iter in range(NUM_ITERATIONS):

    ############################################
    # Find u
    u = cp.Variable(10)
    
    terms = []
    for i in range(X.shape[1]):
        sample_x = X[:, i].reshape(-1,1)
        terms += [-u.T@sample_x + gamma*cp.quad_form(u, Sigma)]
    constraints = [cp.sum(u)==1, u>=-2, u<=2]
    
    problem_1 = cp.Problem(cp.Minimize(cp.sum(terms)), constraints=constraints)
    problem_1.solve()
    u_star = u.value.reshape(-1,1)
    us["Sample"] += [u_star]
    obj_vals["Sample"] += [problem_1.value]

    ############################################
    # Update distribution
    n = X.shape[1]
    X_vars = [cp.Variable((10, 1)) for _ in range(n)]

    terms = []
    for i in range(len(X_vars)):
        terms += [-u_star.T@X_vars[i]]

    constraints = [X_vars[_] >= -1 for _ in range(n)]
    constraints += [X_vars[_] <= 1 for _ in range(n)]
    constraints += [1/n*cp.sum([X_vars[_][0, 0] for _ in range(n)]) - ps[0] <= epsilons[0]]
    constraints += [1/n*cp.sum([X_vars[_][0, 0] for _ in range(n)]) - ps[0] >= -epsilons[0]]
    constraints += [1/n*cp.sum([X_vars[_][1, 0] for _ in range(n)]) - ps[1] <= epsilons[1]]
    constraints += [1/n*cp.sum([X_vars[_][1, 0] for _ in range(n)]) - ps[1] >= -epsilons[1]]

    problem_2 = cp.Problem(cp.Maximize(cp.sum(terms)), constraints=constraints)
    problem_2.solve()
    X = np.hstack([X_vars[_].value.reshape(-1,1) for _ in range(n)])


plt.plot(range(NUM_ITERATIONS), obj_vals["SIP"], label="SIP")
plt.plot(range(NUM_ITERATIONS), obj_vals["Sample"], label="Sample")
plt.legend()
plt.show()

us["SIP"] = np.hstack(us["SIP"])
us["Sample"] = np.hstack(us["Sample"])
plt.plot(us["SIP"][0, :], us["SIP"][1, :], '*', label="SIP")
plt.plot(us["Sample"][0, :], us["Sample"][1, :], '*', label="Sample")
plt.legend()
plt.show()

pdb.set_trace()
plt.plot(X[0, :], X[1, :], '*')
plt.show()