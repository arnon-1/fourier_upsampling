# Fourier Upsampling methods

A small collection of upsampling utilities that lean on Fourier
analysis. I wrote some of them for other projects and decided that it might be useful to share them.

The repository deals with localised upsampling: evaluating a dense slice of a coarsely sampled 1D signal without the need for zero-padding or reconstructing the entire high-resolution grid.
This is similar to FFT with zero padding (Dirichlet kernel interpolation), but implemented more efficiently using the [CZT](https://ccrma.stanford.edu/~jos/st/Bluestein_s_FFT_Algorithm.html).

Several boundary conditions are supported, including periodic, reflective (DCT), and “free” boundaries.
For general upscaling tasks without specific boundary assumptions, `dct_upscale_with_boundaries` is usually the most convenient choice.

## Requirements

* Python
* [NumPy](https://numpy.org/)
* [SciPy](https://scipy.org/)

Install the dependencies with pip:

```bash
python -m pip install numpy scipy
```

## Quick start

```python
import numpy as np
from fourier_upsampling import dirichlet_upscale_zoomfft

# Low-rate signal
x = np.sin(np.linspace(0, 6 * np.pi, 128, endpoint=False))
X = np.fft.fft(x)

# Evaluate indices [400, 600) on the grid that is 8× finer
start, end, q = 400, 600, 8
y = dirichlet_upscale_zoomfft(X, start, end, q).real
```

The FFT-based function assumes the signal is periodic. If your data is more
naturally reflective, use the DCT variant:

```python
from fourier_upsampling import upscale_region_via_dct

y = upscale_region_via_dct(x, start=400, end=600, q=8)
```

For signals with unknown boundary behavior, first blend the edges toward a
constant before upsampling:

```python
from fourier_upsampling import dct_upscale_with_boundaries

safe = dct_upscale_with_boundaries(x, start=400, end=600, q=8)
```

Another option is to fit an overcomplete cosine/sine dictionary and then
evaluate the resulting continuous curve. This approach is considerably more
computationally demanding than the others and seldom offers an advantage over
```dct_upscale_with_boundaries```.
It is the only upscaler in this repository that doesn't fully adhere to the data and tends to dampen higher frequency content.


```python
from fourier_upsampling import free_boundary_upscale

free = free_boundary_upscale(x, start=400, end=600, q=8)
```

Each helper except for ```free_boundary_upscale``` only computes the requested slice, which keeps things quick even for
large upsampling factors.

## Boundary handling overview

* **Periodic** – `dirichlet_upscale_zoomfft` identical to zero padding in the Fourier domain, transforming back and slicing, but more efficient.

* **Reflective** – `upscale_region_via_dct` works in the DCT-II domain,
applying the transform before upsampling. Slightly faster than the above method.

* **Free / blended** – `spectral_upscale` uses periodic boundaries but subtracts a weighted line fit,
periodicises the residual by blending the edges, then restores the trend.  
`dct_upscale_with_boundaries` also uses reflective boundaries but applies
smoothing to the result, which in practice often gives the best outcome.  
  `free_boundary_upscale` solves a ridge-regularised least squares problem in an
overcomplete dictionary of DCT and DST basis functions to smoothly extrapolate the edges. This is relatively slow.


## Examples and comparison plots

`examples.py` visualises the different methods on some demo signals.
It also compares them with linear and quadratic interpolation for reference.
Matplotlib is an additional requirement:

```bash
python examples.py
```
