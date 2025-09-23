import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from numpy.fft import fft

from modules.dct_upscale import upscale_region_via_dct, dct_upscale_with_boundaries
from modules.fft_upscale import spectral_upscale, dirichlet_upscale_zoomfft
from modules.free_boundary_upscale import free_boundary_upscale

mpl.use("TkAgg")


def linear_upscale(x, start, end, r):
    """
    Piecewise-linear upscaler using np.interp.
    x : low-rate samples of length Nl
    Produces values on the high-rate grid over [start, end).
    """
    Nl = len(x)
    t_low = np.arange(Nl)
    t_query = np.arange(start, end) / r  # high-res indices mapped to low-res time
    return np.interp(t_query, t_low, x)


def parabolic_upscale(x, start, end, r):
    """
    Quadratic (parabolic) upscaler.
    Prefers SciPy's quadratic interpolation; otherwise uses a 3-point Lagrange fit.
    """
    t = np.arange(start, end) / r
    Nl = len(x)

    from scipy.interpolate import interp1d
    f = interp1d(np.arange(Nl), x, kind="quadratic", bounds_error=False, fill_value="extrapolate")
    return f(t)


def compare_upscalers(true_signal, r, region, upscalers):
    """
    Compare multiple upscaling methods on a specified high-res region.

    Parameters
    ----------
    true_signal : 1D array
        The ground-truth high-resolution signal (length N_hi).
    r : int
        Down/up-sampling ratio. Downsampled signal will be true_signal[::r].
    region : (start, end)
        High-resolution index range [start, end) to evaluate/plot.
    upscalers : dict[str, callable]
        Mapping of method name -> function(x_low, start, end, r) -> y_up (length end-start).

    Behavior
    --------
    - Creates x_low by decimating true_signal by r.
    - Calls each upscaler on (x_low, start, end, r).
    - Plots ground-truth region, each upscaled curve, and original decimated points.
    - Prints RMS error for each method over the requested region.
    """
    start, end = region
    N_hi = len(true_signal)
    assert 0 <= start < end <= N_hi, "Region must be within the high-res signal."
    assert r >= 2 and N_hi % r == 0, "For simple alignment, N_hi should be divisible by r."

    # Downsample
    x_low = true_signal[::r]

    # Prepare ground-truth slice for the target region
    y_true = true_signal[start:end]
    hi_idx = np.arange(start, end)

    # Run methods
    results = {}
    for name, fn in upscalers.items():
        try:
            y_up = fn(x_low, start, end, r)
            if len(y_up) != (end - start):
                raise ValueError(f"{name} returned length {len(y_up)} (expected {end - start}).")
            results[name] = np.asarray(y_up)
        except Exception as e:
            print(f"[WARN] '{name}' failed: {e}")

    # RMS errors
    print("RMS error on region", region)
    for name, y_up in results.items():
        rms = np.sqrt(np.mean((y_up - y_true) ** 2))
        print(f"  {name:>28s}: {rms:.6g}")

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(len(true_signal)), true_signal, "-o", ms=2, lw=1, label="true (full)")
    for name, y_up in results.items():
        plt.plot(hi_idx, y_up, "-o", ms=3, lw=1.2, label=name)
    # Show original decimated samples aligned to high-res indices
    plt.plot(np.arange(0, N_hi, r), x_low, "o", ms=4, lw=1.2, label="original (downsampled)")
    plt.xlim(start, end - 1)
    plt.legend()
    plt.xlabel("sample (high-res index)")
    plt.ylabel("amplitude")
    plt.title(f"Upscaling comparison on region [{start}, {end}) with r={r}")
    plt.tight_layout()
    plt.show()


upscalers = {
    "DCT upscaled": lambda x, s, e, rr: upscale_region_via_dct(x, s, e, rr),
    "DCT upscaled b_agnostic": lambda x, s, e, rr: dct_upscale_with_boundaries(x, s, e, rr),
    "fft upscaled": lambda x, s, e, rr: dirichlet_upscale_zoomfft(fft(x), s, e, rr).real,
    "fft upscaled b_agnostic": lambda x, s, e, rr: spectral_upscale(x, s, e, rr),
    "free boundary": lambda x, s, e, rr: free_boundary_upscale(x, s, e, rr),
    "linear (np.interp)": lambda x, s, e, r: linear_upscale(x, s, e, r),
    "parabolic (quadratic)": lambda x, s, e, r: parabolic_upscale(x, s, e, r),
}

# =========================
# Example usage / signals
# =========================

# Example 1: Single-tone sine with a non-integer frequency (your original)
true_up = np.sin(np.linspace(-np.pi, np.pi, 1000, endpoint=False) * 17.56587425)
r = 10
region = (100, 900)

compare_upscalers(true_up, r, region, upscalers)

# Example 2: Linear chirp (broadband test)
# (Keeps things minimal and NumPy/Scipy-only; if you have scipy.signal.chirp available, feel free to swap it in.)
N_hi = 2000
t = np.linspace(0, 1, N_hi, endpoint=False)
# simple "chirp-like" sweep by varying phase quadratically
phase = 2 * np.pi * (5 * t + 45 * (t ** 2))  # ~5 Hz to ~95 Hz-ish over [0,1)
true_chirp = np.sin(phase)

r2 = 8
region2 = (300, 1700)
compare_upscalers(true_chirp, r2, region2, upscalers)


def demo_signals(N):
    """
    Return a set of simple demo signals of length N in [0,1).

    Signals
    -------
    s1 : Gaussian bump + linear ramp
    s2 : Mixture of sinusoids + quadratic trend
    s3 : Localized smooth "bump" window + low-frequency sine
    s4 : parabola
    """
    t = np.arange(N) / N

    # 1. Gaussian bump + linear ramp
    s1 = np.exp(-0.5 * ((t - 0.32) / 0.07) ** 2) + 0.25 * t

    # 2. Mixture of oscillations + quadratic
    s2 = np.sin(2 * np.pi * 2.3 * t) \
         + 0.3 * np.cos(2 * np.pi * 0.7 * t) \
         + 0.1 * (t - 0.5) ** 2

    # 3. Smooth "bump" via polynomial smoothstep window
    def smoothstep(z):
        return np.where(
            (z >= 0) & (z <= 1),
            6 * z ** 5 - 15 * z ** 4 + 10 * z ** 3,
            np.where(z > 1, 1.0, 0.0)
        )

    center, width = 0.65, 0.25
    z = (t - (center - width / 2)) / width
    window = smoothstep(z) * (1 - smoothstep(z))
    s3 = 0.9 * window + 0.05 * np.sin(2 * np.pi * 0.15 * t)

    return [s1, s2, s3, ((t-0.4)*8)**2, np.exp((t-0.4)*10)]


signals = demo_signals(2000)
names = ["Gaussian+ramp", "sinusoidal+quad", "smooth bump", "parabola", "exponential"]

for sig, name in zip(signals, names):
    print(f"\n=== {name} ===")
    compare_upscalers(sig, r=50, region=(300, 1700), upscalers=upscalers)
