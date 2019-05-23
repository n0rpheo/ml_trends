import cvxpy as cp
import numpy as np


def test_func(x, l):

     intersection = 0
     union = 0
     for xv, lv in zip(x, l):
          intersection += xv*lv
          union += (xv+lv) - (xv*lv)
     return cp.sum(union)


n = 2    # number of mentions
m = 1     # number of clusters

# Create two scalar optimization variables.
x = cp.Variable(3, boolean=True)

# Create two constraints.
constraints = [sum(x) <= 2,
               -sum(x) <= -1]

# Form objective.

ll = [1, 1, 1]

obj = cp.Minimize(sum([-test_func(x, ll)*i for i in range(5)]))

# Form and solve problem.
prob = cp.Problem(obj, constraints)
prob.solve()  # Returns the optimal value.
print("status:", prob.status)
print("optimal value", prob.value)
print("optimal var", [round(xv.value, 0) for xv in x])