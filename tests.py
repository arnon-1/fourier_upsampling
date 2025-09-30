import numpy as np
from numpy.fft import fft

from fourier_upsampling import (
    dct_upscale_with_boundaries,
    dirichlet_upscale_zoomfft,
    spectral_upscale,
    upscale_region_via_dct,
)


def _dirichlet_upscale_reference(x, start, end, q):
    """
    Reference implementation via zero-padding the DFT to length N*q
    (with correct Nyquist handling for even N), then IFFT and slice.

    Returns x_up[start:end].
    """
    N = x.shape[0]
    assert isinstance(start, int) and isinstance(end, int), "start/end must be ints"
    assert 0 <= start < end <= N * q, "slice must lie within [0, N*q]"

    if q == 1:
        # No interpolation needed; identity.
        return x[start:end]

    Nq = N * q
    X = np.fft.fft(x, n=N)
    Y = np.zeros(Nq, dtype=X.dtype)

    if N % 2 == 0:
        K = N // 2
        Y[:K] = X[:K]
        ny = X[K].real
        Y[K] = ny / 2.0
        Y[-K] = ny / 2.0
        Y[-(N - K - 1):] = X[K + 1:]
    else:
        K = (N + 1) // 2
        Y[:K] = X[:K]
        Y[-(N - K):] = X[K:]

    x_up = q * np.fft.ifft(Y, n=Nq)

    # Sanity check: downsample recovers original samples (global property)
    diff = x_up[::q] - x
    rel = np.linalg.norm(diff) / (np.linalg.norm(x) + 1e-15)
    assert rel < 6e-10, (len(x), q, rel)

    return x_up[start:end]


def test_matches_reference_many_cases():
    rng = np.random.default_rng(1234)

    # A few fixed, diverse cases (odd/even N, short/long windows, misaligned/aligned)
    fixed = [
        (64, 3, 0, 96),  # even N, from start
        (63, 5, 10, 110),  # odd N, unaligned start/length
        (128, 4, 256, 256 + 120),  # even N, mid segment
        (75, 7, 0, 140),  # odd N, from start
        (81, 9, 700, 729),  # odd N, non-multiple-of-q length
    ]

    # Random cases (downscaled vs original to limit compute)
    random_cases = []
    for _ in range(60):
        N = int(rng.integers(10, 1000))
        q = int(rng.integers(1, max(2, N - 1)))
        Nq = N * q
        Mmax = min(Nq, 2000)
        M = int(rng.integers(1, Mmax + 1))
        start = int(rng.integers(0, Nq - M + 1))
        end = start + M
        random_cases.append((N, q, start, end))

    # Extra aligned cases so that y[::q] should match a contiguous slice of x
    for _ in range(20):
        N = int(rng.integers(10, 500))
        q = int(rng.integers(1, max(2, N - 1)))
        L = int(rng.integers(1, min(N, 500)))  # number of original samples to cover
        start = q * int(rng.integers(0, N - L + 1))
        end = start + q * L
        fixed.append((N, q, start, end))

    cases = fixed + random_cases

    for i, (N, q, start, end) in enumerate(cases):
        x = rng.standard_normal(N)
        Fe = fft(x)

        y_czt = dirichlet_upscale_zoomfft(Fe, start, end, q)
        y_ref = _dirichlet_upscale_reference(x, start, end, q)

        assert len(y_czt) == (end - start), (N, q, start, end, "length mismatch")

        rel = np.linalg.norm(y_czt - y_ref) / (np.linalg.norm(y_ref) + 1e-15)
        assert rel < 2e-8, (i, N, q, start, end, rel, "doesn't match reference calculation")

        # Where the upsampled indices coincide with original-grid samples
        aligned_start = start + ((q - start % q) if start % q else 0)
        aligned_end = (end - 1) - (end - 1) % q
        Lslice = (aligned_end - aligned_start) // q + 1
        lo = aligned_start // q
        lowres_diff_ref = y_ref[aligned_start - start::q] - x[lo: lo + Lslice]
        relr = np.linalg.norm(lowres_diff_ref) / (np.linalg.norm(y_ref) + 1e-15)
        assert relr < 6e-10, (i, N, q, start, end, relr, "reference doesn't match original at known points")

        lowres_diff_czt = y_czt[aligned_start - start::q] - x[lo: lo + Lslice]
        relc = np.linalg.norm(lowres_diff_czt) / (np.linalg.norm(y_czt) + 1e-15)
        assert relc < 6e-10, (i, N, q, aligned_start, end, relc, "zoomfft doesn't match original at known points")

        # Results should be (numerically) real for real x
        assert np.max(np.abs(np.imag(y_czt))) < 2e-8, (i, N, q, start, end, "should be real")
        if min(start, N * q - end) // q == 0:
            continue

        y_periodic_upscale = spectral_upscale(x, start, end, q)
        lowres_diff = y_periodic_upscale[aligned_start - start::q] - x[lo: lo + Lslice]
        relc = np.linalg.norm(lowres_diff) / (np.linalg.norm(y_periodic_upscale) + 1e-15)
        assert relc < 6e-10, (i, N, q, aligned_start, end, relc, "periodic upscale doesn't match original at known "
                                                                 "points")

        y_dct_upscale = upscale_region_via_dct(x, start, end, q)
        lowres_diff = y_dct_upscale[aligned_start - start::q] - x[lo: lo + Lslice]
        relc = np.linalg.norm(lowres_diff) / (np.linalg.norm(y_dct_upscale) + 1e-15)
        assert relc < 6e-10, (i, N, q, aligned_start, end, relc, "dct upscale doesn't match original at known "
                                                                 "points")

        try:
            y_dct_upscale = dct_upscale_with_boundaries(x, start, end, q)
        except ValueError as e:
            # The boundary-aware variant requires extra guard samples; skip cases
            # where the requested window is too close to the edges.
            raise e
            continue

        lowres_diff = y_dct_upscale[aligned_start - start::q] - x[lo: lo + Lslice]
        relc = np.linalg.norm(lowres_diff) / (np.linalg.norm(y_dct_upscale) + 1e-15)
        assert relc < 6e-10, (i, N, q, aligned_start, end, relc, "dct boundary-agnostic upscale doesn't match "
                                                                 "original at known points")


def test_constant_signals():  # TODO: other functions as well
    rng = np.random.default_rng(0)
    cases = [
        (32, 4, 0, 64),
        (63, 3, 10, 160),
        (128, 5, 100, 100 + 333),
        (75, 2, 50, 150),
    ]
    for _ in range(20):
        N = int(rng.integers(8, 300))
        q = int(rng.integers(1, max(2, N - 1)))
        Nq = N * q
        M = int(rng.integers(1, min(Nq, 1000)))
        # Guard against edge cases by clamping end within [0, Nq]
        start = int(rng.integers(0, max(1, Nq - M + 1)))
        end = start + M
        if end > Nq:
            end = Nq
            start = max(0, end - M)
        cases.append((N, q, start, end))
    for (N, q, start, end) in cases:
        c = rng.standard_normal()
        x = np.full(N, c)
        y = dirichlet_upscale_zoomfft(fft(x), start, end, q)
        err = np.max(np.abs(y.real - c))
        assert err < 5e-10, (N, q, start, end, err, "constant not preserved")
        assert np.max(np.abs(y.imag)) < 2e-8, (N, q, start, end, "should be real")


if __name__ == "__main__":
    test_constant_signals()
    test_matches_reference_many_cases()
    print("All tests passed.")
