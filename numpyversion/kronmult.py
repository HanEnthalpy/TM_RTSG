import numpy as np


def kronmult(Q, X):
    """
    KRONMULT Efficient Kronecker Multiplication

    Y=kronmult(Q,X) computes
        Y = (Q[0] ⊗ Q[1] ⊗ ... ⊗ Q[m-1]) * X
    without forming the full Kronecker product matrix.

    Parameters:
        Q: list of 2D arrays (square matrices)
        X: 2D array with shape (prod([q.shape[0] for q in Q]), k)

    Returns:
        Y: result of Kronecker product multiplication

    Reference:
        Fernandes et al. 1998, JACM 45(3): 381-414
        doi:10.1145/278298.278303
    """

    N = len(Q)
    n = np.zeros(N, dtype=int)
    nleft = 1
    nright = 1

    for i in range(N - 1):
        n[i] = Q[i].shape[0]
        nleft *= n[i]

    if N > 0:
        n[N - 1] = Q[N - 1].shape[0]

    for i_idx in range(N - 1, -1, -1):
        base = 0
        jump = n[i_idx] * nright

        for k in range(nleft):
            for j in range(nright):
                start_index = base + j

                stop_index = start_index + nright * (n[i_idx] - 1) + 1

                block = X[start_index:stop_index:nright, :]
                new_block = Q[i_idx] @ block
                X[start_index:stop_index:nright, :] = new_block
            base += jump

        if i_idx > 0:
            nleft //= n[i_idx - 1]
        else:
            nleft //= n[0]
        nright *= n[i_idx]

    return X