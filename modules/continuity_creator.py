import numpy as np
from scipy.signal.windows import kaiser


def _monotone_taper(m, beta=8.6):
    """
    Monotone increasing weights w in [0,1] of length m using a cumsummed
    Kaiser window, normalized so w[0]=0, w[-1]=1.
    """
    if m < 1:
        raise ValueError("m must be >= 1")
    w0 = kaiser(m, beta=beta).astype(float)
    w = np.cumsum(w0)
    w /= w[-1]
    w[0] = 0.0
    w[-1] = 1.0
    return w


def make_periodic(x, m, beta=8.6):
    """
    Make a 1D array x periodic by replacing both boundary regions of length m
    with the same smoothly blended edge segment via a partition-of-unity crossfade.

    Parameters
    ----------
    x : (N,) array_like
        Input signal (real or complex).
    m : int
        Length of boundary region at each end (1 <= m <= N//2).
    beta : float, optional
        Kaiser beta for taper smoothness (default 8.6).

    Returns
    -------
    y : (N,) ndarray
        Output signal equal to x in the interior, with identical blended segments
        written into both edges [0:m) and [N-m:N) to ensure periodicity.
    """
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError("x must be 1D")
    N = x.size
    if m < 1 or m > N // 2:
        raise ValueError("m must satisfy 1 <= m <= N//2")

    w = _monotone_taper(m, beta=beta)  # increases 0->1
    v = 1.0 - w

    L = x[:m]
    R = x[-m:]
    blended = w * L + v * R

    y = x.copy()
    y[:m] = blended
    return y[:-m]


def _weighted_line_fit(x, w):
    """
    Weighted LS fit of x[n] ≈ a*n + b where weights live only on the edges.
    `w` is the left-edge half-window of length m; the right edge mirrors it.
    The middle (N-2*m) has zero weight.
    Returns complex a, b if x is complex (design matrix is real).
    """
    x = np.asarray(x)
    w = np.asarray(w, dtype=float)
    N = x.shape[0]
    m = int(len(w))

    if m < 0 or 2 * m > N:
        raise ValueError("len(w) must satisfy 2*len(w) <= len(x)")

    # indices
    n = np.arange(N, dtype=float)

    # build full weights: left edge = w, right edge = reversed w, middle = 0
    W = np.zeros(N, dtype=float)
    if m > 0:
        W[:m] = w
        W[-m:] = w[::-1]

    # weighted sums
    S0 = W.sum()
    S1 = (W * n).sum()
    S2 = (W * n * n).sum()
    T0 = (W * x).sum()
    T1 = (W * n * x).sum()

    det = S2 * S0 - S1 * S1
    if det == 0:
        raise np.linalg.LinAlgError("Singular system: check your weights.")

    a = (S0 * T1 - S1 * T0) / det
    b = (S2 * T0 - S1 * T1) / det
    return a, b


def test_weighted_line_fit():
    from numpy.linalg import lstsq

    N = 200
    n = np.arange(N)
    true_a, true_b = 0.7, -3.0
    rng = np.random.default_rng(0)
    x = true_a * n + true_b + 0.05 * rng.standard_normal(N)

    # ordinary (unweighted) LS for reference
    A = np.vstack([n, np.ones_like(n)]).T
    a_ref, b_ref = lstsq(A, x, rcond=None)[0]

    # edge-weighted fit (edges count, middle doesn't)
    m = 40
    w = np.ones(m)
    a, b = _weighted_line_fit(x, w)

    # sanity checks: unweighted LS should be close to ground truth
    assert np.allclose([a_ref, b_ref], [true_a, true_b], atol=1e-2)

    # edge-weighted LS should still roughly capture slope and intercept
    assert np.isfinite(a) and np.isfinite(b)
    assert abs(a - true_a) < 0.1
    assert abs(b - true_b) < 0.5


def fit_line_on_edges(x, boundary_width, beta=8.6):
    k = kaiser(2 * boundary_width, beta=beta).astype(float)
    k /= k.max()
    w = np.zeros(len(x))
    w[:boundary_width] = k[-boundary_width:]
    w[-boundary_width:] = k[:boundary_width]
    return _weighted_line_fit(x, k[-boundary_width:])


def add_line_on_upsampled_grid(a, b, start, end, q):
    k = np.arange(start, end, dtype=float)
    # n_up = k/q, so line(n_up) = a*(k/q) + b
    return (a * (k / q)) + b


if __name__ == "__main__":
    test_weighted_line_fit()
    from matplotlib import pyplot as plt

    #plt.plot(_monotone_taper(50))
    plt.plot(np.concatenate([make_periodic(np.cos(np.linspace(-100, 100, 2000)), m=1000, beta=np.sqrt(1000))] * 2))
    plt.show()
