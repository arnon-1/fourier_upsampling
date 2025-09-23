import numpy as np
from scipy.signal import czt
from scipy.fft import dct


def reconstruct_region_from_coeffs(C, N, start, end, r):
    """
    Fast reconstruction on a dense grid in the *time* window [start, end],
    evaluated at t = start, start+1/r, ..., end   (length (end-start)*r + 1).

    Assumes C are DCT-II coefficients computed with norm='ortho'.
    """
    assert 0 <= start < end < N * r
    end -= 1  # exclusive -> inclusive
    assert r >= 1 and int(r) == r
    r = int(r)

    # weights for orthonormal DCT-II basis: alpha_0 = 1/sqrt(2), alpha_k = 1
    Xk = C.copy()
    Xk[0] /= np.sqrt(2.0)

    # CZT over k to evaluate the cosine series at t = start + m/r
    M = (end - start) + 1
    A = np.exp(-1j * (np.pi / N) * (start / r + 0.5))  # start of t
    W = np.exp(1j * (np.pi / N) * (1.0 / r))  # step of 1/r in t
    S = czt(Xk, m=M, w=W, a=A)

    # overall scale for orthonormal DCT basis
    return np.sqrt(2.0 / N) * np.real(S)


def upscale_region_via_dct(x, start, end, r):
    """
    Upscale only the time-domain region [start, end] by factor r using a DCT-II model.
    So it assumes reflective boundaries.
    Returns samples at t = start..end in 1/r steps.
    """
    x = np.asarray(x, float)
    N = x.size
    # SciPy DCT-II with orthonormal scaling
    C = dct(x, type=2, norm='ortho')
    return reconstruct_region_from_coeffs(C, N, start, end, r)


def dct_upscale_with_boundaries(
        x, start, end, q,
        *,
        boundary_samples=None,  # int or (mL, mR)
        beta=None,  # float or (betaL, betaR)
):
    """
    Wrapper for DCT upscaling, makes the signal
    boundary-agnostic by smoothly tapering each end toward a constant
    (estimated separately for left and right) before DCT-based upscaling.
    Do not use this if the signal has actual reflective boundaries.

    Parameters
    ----------
    x : array_like
        1D signal on the base grid (length N).
    start, end : int
        Requested slice [start, end) on the upsampled (N*q) grid.
    q : int
        Integer upsampling factor.
    boundary_samples : int or (int, int), optional
        Base-grid width(s) of the blended boundary regions. If None, a
        per-side heuristic is used based on distance to that side.
    beta : float or (float, float), optional
        Kaiser beta per side. If None, defaults to sqrt(m_side) per side.

    Returns
    -------
    y : ndarray
        Samples on the requested upsampled slice.

    Notes
    -----
    - Each side’s constant is computed as mean((1-w)*x_side)/mean(1-w).
    - Uses half a Kaiser window per side; no cumulative sums.
    """
    x = np.asarray(x, float)
    N = x.size
    q = int(q)
    if q <= 0:
        raise ValueError("q must be a positive integer")

    # Free space to each upsampled boundary
    free_left_up = int(start-q)
    free_right_up = int((N-1) * q - end)
    if free_left_up <= 0 or free_right_up <= 0:
        raise ValueError("too close to boundary")

    # Convert to base-grid samples available on each side
    free_left_base = free_left_up // q
    free_right_base = free_right_up // q

    # Resolve mL, mR
    if boundary_samples is None:
        mL = int(np.sqrt(max(free_left_base, 0) * 3))
        mR = int(np.sqrt(max(free_right_base, 0) * 3))
    else:
        if np.isscalar(boundary_samples):
            mL = mR = int(boundary_samples)
        else:
            mL, mR = map(int, boundary_samples)

    # Validate per side
    if mL <= 0 or mR <= 0:
        raise ValueError("too close to boundary")
    if mL > free_left_base:
        raise ValueError("too many samples in left boundary layer")
    if mR > free_right_base:
        raise ValueError("too many samples in right boundary layer")

    # Resolve per-side beta
    if beta is None:
        betaL = float(np.sqrt(mL))
        betaR = float(np.sqrt(mR))
    else:
        if np.isscalar(beta):
            betaL = betaR = float(beta)
        else:
            betaL, betaR = map(float, beta)

    # Build half-Kaiser tapers per side
    fullL = np.kaiser(2 * mL, betaL)
    wL = fullL[:mL]  # 0 -> 1 over left boundary
    fullR = np.kaiser(2 * mR, betaR)
    wR = fullR[:mR][::-1]  # 1 -> 0 over right boundary when blending x[-mR:]

    # Per-side constants from the 'left-out' parts
    one_minus_wL = 1.0 - wL
    denomL = one_minus_wL.sum()
    cL = (one_minus_wL * x[:mL]).sum() / denomL if denomL > 0 else float(x[0])

    one_minus_wR = 1.0 - wR
    denomR = one_minus_wR.sum()
    xR = x[-mR:]
    cR = (one_minus_wR * xR).sum() / denomR if denomR > 0 else float(x[-1])

    # Blend boundaries independently
    x_mod = x.copy()
    x_mod[:mL] = wL * x[:mL] + (1.0 - wL) * cL
    x_mod[-mR:] = wR * x[-mR:] + (1.0 - wR) * cR

    # Reflective DCT upscaling on the blended signal
    return upscale_region_via_dct(x_mod, int(start), int(end), q)


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    import matplotlib as mpl
    mpl.use("TkAgg")

    true_up = np.sin(np.linspace(-np.pi, np.pi, 1000, endpoint=False) * 17.56587425)
    x = np.sin(np.linspace(-np.pi, np.pi, 100, endpoint=False) * 17.56587425)

    # Upscale region [250, 750] by r=10 (i.e., evaluate every 0.1 sample)
    start_end = 100, 900
    x_up = upscale_region_via_dct(x, *start_end, 10)
    x_up2 = dct_upscale_with_boundaries(x, *start_end, 10)

    plt.plot(np.arange(1000), true_up, "-o", label="true")
    plt.plot(np.arange(*start_end), x_up, "-o", label="DCT upscaled")
    plt.plot(np.arange(*start_end), x_up2, "-o", label="DCT upscaled b_agnostic")
    plt.plot(np.arange(1000)[::10], x, "-o", label="original (downsampled)")
    plt.legend()
    plt.show()

    true_up = np.sin(np.linspace(-np.pi, np.pi, 100, endpoint=False))
    x = np.sin(np.linspace(-np.pi, np.pi, 11, endpoint=True))

    # Upscale region [250, 750] by r=10 (i.e., evaluate every 0.1 sample)
    x_up = upscale_region_via_dct(x, 10, 90, 10)
    # Upscale region [250, 750] by r=10 (i.e., evaluate every 0.1 sample)
    x_up2 = dct_upscale_with_boundaries(x, 10, 90, 10)

    print(np.allclose(x_up, x_up2))

    plt.plot(np.arange(100), true_up, "-o", label="true")
    plt.plot(np.arange(10, 90), x_up, "-o", label="DCT upscaled")
    plt.plot(np.arange(10, 90), x_up2, "-o", label="DCT upscaled b_agnostic")
    plt.plot(np.linspace(0, 100, 11, endpoint=True), x, "o", label="original (downsampled)")
    plt.legend()
    plt.show()
