import numpy as np
from scipy.signal import czt

from modules.continuity_creator import make_periodic, fit_line_on_edges, add_line_on_upsampled_grid


def dirichlet_upscale_zoomfft(X, start, end, q):
    """
    Dirichlet (zero-padded) interpolation over an arbitrary window
    [start, end) on the length-N*q upsampled grid, using a CZT-based
    evaluation.

    Parameters
    ----------
    X : array_like
        Length-N DFT of the time-domain x (numpy.fft.fft convention).
    start, end : int
        Slice bounds in the upsampled domain (0 <= start < end <= N*q).
    q : int
        Integer upsampling factor.

    Returns
    -------
    y : ndarray, shape (end-start,)
        x_up[start:end], where x_up is the Dirichlet-interpolated signal
        obtained by zero-padding the spectrum to length N*q.
    """
    N = X.shape[0]
    assert isinstance(start, int) and isinstance(end, int), "start/end must be ints"
    assert 0 <= start < end <= N * q, "slice must lie within [0, N*q]"

    M = end - start
    Nq = N * q

    # Base CZT step per sample on the N*q grid
    B = np.exp(1j * 2 * np.pi / Nq)  # one step along the N*q root-of-unity circle
    A = B ** (-start)  # starting point at index 'start'

    if N % 2 == 0:
        # even N: split out Nyquist bin explicitly
        K = N // 2
        # DC..(K-1)
        Y1 = czt(X[:K], M, B, A)
        # (K+1)..(N-1) (negative freqs excluding Nyquist)
        Y2 = czt(X[K + 1:], M, B, A)

        k = np.arange(M)
        zeta = np.exp(-1j * 2 * np.pi * (N - K - 1) * (start + k) / Nq)

        ny = X[K].real  # Nyquist correction term
        cos_phase = np.cos(2 * np.pi * (start + k) * K / Nq)

        y = (Y1 + zeta * Y2 + ny * cos_phase) / N
    else:
        # odd N: no Nyquist bin
        K = (N + 1) // 2
        Y1 = czt(X[:K], M, B, A)
        Y2 = czt(X[K:], M, B, A)

        k = np.arange(M)
        zeta = np.exp(-1j * 2 * np.pi * (N - K) * (start + k) / Nq)
        y = (Y1 + zeta * Y2) / N

    return y


def spectral_upscale(
        x, start, end, q,
        *,
        boundary_samples=None,
        beta=None,
):
    """
    Wrapper around dirichlet_upscale_zoomfft that periodicises x before upscaling.
    Automatically handles edges by smoothly blending them
     but can give underwhelming results when upscaling close to these edges.
    Do not use this when the signal is already periodic.

    Parameters
    ----------
    x : array_like
        The array to upscale.
    start, end : int
        Requested slice [start, end) on length N*q grid.
    q : int
        Integer upsampling factor.
    boundary_samples : int
        Number of samples in the upscaled boundary layer. Chooses a sensible default when left None.
    beta : float
        Kaiser parameter controlling the width of the main lobe. Chooses a sensible default when left None.
    """
    N = x.shape[0]
    # Free space in the upsampled domain
    free_space_up = min(start, N * q - end)

    # Convert to base-grid samples and pick m
    free_space_base = free_space_up // q
    if boundary_samples is None:
        # Heuristic for m: grow sublinearly with distance
        m = int(np.sqrt(free_space_base * 3))
    else:
        if boundary_samples > free_space_base:
            raise ValueError("too many samples in boundary layer")
        else:
            m = boundary_samples
    if m == 0:
        raise ValueError("too close to boundary")

    # Choose beta
    beta = float(np.sqrt(m)) if beta is None else beta

    a, b = fit_line_on_edges(x, m, beta=beta)
    r = x - (a * np.arange(N) + b)
    # Time-domain periodicise: IFFT -> make_periodic -> FFT
    r_per = make_periodic(r, m, beta=beta)
    R_per = np.fft.fft(r_per)

    r_up = dirichlet_upscale_zoomfft(R_per, start, end, q).real
    x_up = r_up + add_line_on_upsampled_grid(a, b, start, end, q)
    return x_up


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    true_up = np.sin(np.linspace(-np.pi, np.pi, 1001) / 2)
    x = np.sin(np.linspace(-np.pi, np.pi, 101) / 2)
    x_up = spectral_upscale(x, 250, 750, 10)
    plt.plot(np.arange(1001), true_up, "-o")
    plt.plot(np.arange(250, 750), x_up,  "-o")
    plt.plot(np.arange(1001)[::10], x, "o")
    plt.show()


def refine_peak_dirichlet(x, peak_idx, q=16, symmetric_extend=False):
    """
    Refine a detected peak location by Dirichlet (zero-padded) interpolation
    over the range [peak_idx-1, peak_idx+1] on an upsampled grid.

    Parameters
    ----------
    x : array_like
        Input 1D array (low-resolution samples).
    peak_idx : int
        Detected peak index in `x`.
    q : int, optional
        Integer upsampling factor (default: 16).
    symmetric_extend : bool, optional
        If True, mirror-extend the array as [x[::-1], x] before
        interpolation to reduce boundary artifacts (default: False).

    Returns
    -------
    idx_refined : float
        Sub-sample peak index referred to the original `x` indexing.
    val_refined : float or complex
        Interpolated peak value at `idx_refined`.
    """
    x = np.asarray(x)
    N = x.size

    if symmetric_extend:
        # Mirror on both sides once; original segment starts at offset N
        x_ext = np.concatenate([x[::-1], x])
        offset = N
    else:
        x_ext = x
        offset = 0

    N_ext = x_ext.size
    # Peak index in the (possibly) extended signal
    i_ext = int(peak_idx) + offset
    # Ensure we can take a 1-step neighborhood on both sides
    i_ext = int(np.clip(i_ext, 1, N_ext - 2))

    # FFT of the (possibly) extended signal
    X = np.fft.fft(x_ext)

    # On the length-(N_ext*q) grid, original sample k maps to k*q
    start = max(0, (i_ext - 1) * q)
    end = min(N_ext * q, (i_ext + 2) * q)  # exclusive

    # Upsample only the local slice and locate the sub-sample maximum
    y_up = dirichlet_upscale_zoomfft(X, start, end, q)
    y_up = np.real_if_close(y_up)  # keep real if numerically real
    i_rel = int(np.argmax(y_up))

    idx_up_global = start + i_rel  # index on upsampled grid
    idx_ext_float = idx_up_global / q  # float index on extended grid
    idx_refined = idx_ext_float - offset  # back to original x indexing
    val_refined = y_up[i_rel]

    return idx_refined, val_refined
