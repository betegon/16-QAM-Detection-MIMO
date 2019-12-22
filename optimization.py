import cvxpy as cp
import numpy as np


def min_tr_WA(k,A):
    """ Minizime trace(W*A), equations stated in paper "Semidefinite Relaxation
        for Detection of 16-QAM Signaling in MIMO Channels".

    Args:
        k  (int): Number of antennas.
        A  (np.ndarray): Matrix needed to formulate problem to minimize.

    Returns:
        W (np.ndarray): Optimal W matrix.

    """
    # Problem data.
    np.random.seed(1)
    zero_array = np.zeros(2*k)

    # Construct the problem.
    W = cp.Variable((4*k+1,4*k+1), PSD=True)
    objective = cp.Minimize(cp.trace(cp.matmul(W,A)))
    constraints = [
        W >> 0,
        cp.diag(W[0:2*k, 0:2*k]) - W[2*k:4*k,4*k] == 0,
        cp.diag(W[2*k:4*k, 2*k:4*k]) - 10*W[2*k:4*k,4*k] + 9*np.ones((2*k)) == 0,
        W[4*k, 4*k] == 1]

    # formulate and solve problem
    prob = cp.Problem(objective, constraints)
    prob.solve()

    # print(W.value) # A numpy ndarray.
    return W.value
