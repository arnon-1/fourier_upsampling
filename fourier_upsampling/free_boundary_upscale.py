# note that I used chatgpt a lot for this file (possible mistakes are not my fault)

import numpy as np


def build_dictionary_at(t, N, Kc=None, Ks=None, include_const=True, include_linear=True):
    """
    Build an overcomplete trigonometric + low-order polynomial dictionary.

    Basis:
      - cos(pi * (t + 0.5) * k / N), k = 0..Kc-1 (k=0 scaled by 1/sqrt(2))
      - sin(pi * (t + 0.5) * l / N), l = 1..Ks
      - optional: constant term (1)
      - optional: linear term ((t - (N-1)/2)/N)

    Parameters
    ----------
    t : array_like
        Sample points (can be non-integers for continuous evaluation).
    N : int
        Nominal period length (also used to scale frequencies).
    Kc : int or None
        Number of cosine terms (default: N).
    Ks : int or None
        Number of sine terms (default: N-1).
    include_const, include_linear : bool
        Include constant / linear columns.

    Returns
    -------
    A : ndarray, shape (len(t), n_cols)
        Design matrix (dictionary) evaluated at t.
    meta : dict
        Dictionary parameters needed to rebuild the basis later.
    """
    t = np.asarray(t, dtype=float).ravel()
    Kc = N if Kc is None else Kc
    Ks = (N - 1) if Ks is None else Ks

    # Frequencies (broadcast-ready)
    arg = np.pi * (t[:, None] + 0.5) / N

    # Cos block (k = 0..Kc-1), with k=0 scaled by 1/sqrt(2)
    kc = np.arange(Kc)
    cos_block = np.cos(arg * kc)
    if Kc > 0:
        cos_block[:, 0] /= np.sqrt(2.0)

    # Sin block (l = 1..Ks)
    ks = np.arange(1, Ks + 1)
    sin_block = np.sin(arg * ks) if Ks > 0 else np.empty((t.size, 0))

    cols = [cos_block, sin_block]

    if include_const:
        cols.append(np.ones((t.size, 1), dtype=float))
    if include_linear:
        cols.append(((t - (N - 1) / 2.0) / N)[:, None])

    A = np.hstack(cols)
    meta = dict(N=N, Kc=Kc, Ks=Ks, include_const=include_const, include_linear=include_linear)
    return A, meta


def frequency_weights(N, Kc, Ks, include_const=True, include_linear=True, p=4):
    """
    Per-coefficient frequency weights (ω^p), with zeros for DC/const/linear.

    Returns
    -------
    w : ndarray, shape (n_cols,)
        Weights aligned with columns of the dictionary built by build_dictionary_at().
    """
    # Cosine k = 0..Kc-1  (set k=0 weight to 0)
    k = np.arange(Kc)
    w_cos = (np.pi * k / N) ** p
    if Kc > 0:
        w_cos[0] = 0.0

    # Sine l = 1..Ks
    l = np.arange(1, Ks + 1)
    w_sin = (np.pi * l / N) ** p

    extra = []
    if include_const:
        extra.append(0.0)
    if include_linear:
        extra.append(0.0)

    return np.concatenate([w_cos, w_sin, np.array(extra, dtype=float)])


def fit_overcomplete(x, lam=1e-1, p=4, Kc=None, Ks=None, include_const=True, include_linear=True):
    """
    Ridge-regularized least-squares fit of x to the overcomplete dictionary.

    Solves: (A^T A + λ diag(w^2)) c = A^T x, where w are frequency weights (ω^p).

    Returns
    -------
    c : ndarray
        Coefficients of the fit.
    y : ndarray
        In-sample reconstruction A @ c.
    meta : dict
        Basis parameters for later evaluation.
    """
    x = np.asarray(x, dtype=float).ravel()
    N = x.size
    t_disc = np.arange(N, dtype=float)

    A, meta = build_dictionary_at(
        t_disc, N, Kc=Kc, Ks=Ks,
        include_const=include_const, include_linear=include_linear
    )
    w = frequency_weights(
        N, meta["Kc"], meta["Ks"], meta["include_const"], meta["include_linear"], p=p
    )

    # Normal equations with diagonal regularization (avoid forming diag matrix)
    AtA = A.T @ A
    Atb = A.T @ x
    diag_add = lam * (w ** 2)
    AtA.flat[::AtA.shape[0] + 1] += diag_add  # add to diagonal in-place

    try:
        c = np.linalg.solve(AtA, Atb)
    except np.linalg.LinAlgError:
        c, *_ = np.linalg.lstsq(AtA, Atb, rcond=None)

    y = A @ c
    return c, y, meta


def fit_overcomplete_lowfreq(x, Kc=None, Ks=None,
                             include_const=True, include_linear=True):
    """
    Unregularized least-squares fit using ONLY low-frequency atoms.
    Choose Kc, Ks as small cutoffs so the dictionary is low-pass.
    """
    x = np.asarray(x, dtype=float).ravel()
    N = x.size
    t_disc = np.arange(N, dtype=float)

    A, meta = build_dictionary_at(
        t_disc, N, Kc=Kc, Ks=Ks,
        include_const=include_const, include_linear=include_linear
    )

    c, *_ = np.linalg.lstsq(A, x, rcond=None)
    y = A @ c
    return c, y, meta


def evaluate_continuous(c, eval_t, meta):
    """
    Evaluate the fitted continuous curve at arbitrary points eval_t.
    """
    A_eval, _ = build_dictionary_at(
        eval_t, meta["N"], meta["Kc"], meta["Ks"],
        meta["include_const"], meta["include_linear"]
    )
    return A_eval @ c


def periodic_extension(values_one_period, r, shift=0):
    """
    Repeat a single-period array r times.
    """
    out = np.tile(values_one_period, int(r))
    N = len(values_one_period)
    for i in range(N):
        out[i * N: (i + 1) * N] += shift * i
    return out


def free_boundary_upscale(x, start, end, q, lam=1e-1, p=4, include_linear=False):
    c, _, meta = fit_overcomplete(x, lam=lam, p=p, include_linear=include_linear, include_const=include_linear)
    fine_t = np.arange(start, end)/q
    return evaluate_continuous(c, fine_t, meta)

