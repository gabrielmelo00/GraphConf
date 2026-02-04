import numpy as np

def random_adj_from_gt(A, noise_level=0.3, seed=None):
    """
    Flip edges of adjacency matrix with given probability.
    Keeps symmetry and zero diagonal.
    """
    rng = np.random.default_rng(seed)
    A_noisy = A.copy()

    mask = rng.random(A.shape) < noise_level
    A_noisy[mask] = 1 - A_noisy[mask]

    # enforce symmetry and zero diagonal
    A_noisy = np.triu(A_noisy, 1)
    A_noisy = A_noisy + A_noisy.T
    np.fill_diagonal(A_noisy, 0)

    return A_noisy