# pn_adev_allan

相噪阿伦算法 — Phase Noise & Allan Deviation Algorithms

> **问题说明 / Problem statement**  
> 相噪阿伦算法，阿伦基本正确，相噪算法有问题（pn_adev_closetrue.zip），
> 相噪算法没问题，但是降采样有问题（pn_calc.zip）

This repository provides correct Python implementations of two key
frequency-metrology algorithms, along with a full unit-test suite that
validates both the formulas and the known bug-fixes.

---

## Modules

### `pn_calc.py` — Phase Noise (相噪)

| Function | Description |
|---|---|
| `calc_phase_noise(phase_data, fs, window, nperseg)` | SSB phase noise L(f) \[dBc/Hz\] from a phase time series |
| `decimate_phase(phase_data, fs, decimate_factor)` | Anti-aliased decimation (FIR low-pass filter before downsampling) |
| `calc_phase_noise_multidecade(phase_data, fs, ...)` | Multi-decade PN by cascaded decimation |

**Bugs fixed vs. reference files:**

1. **Phase-noise formula (`pn_adev_closetrue.zip`)**  
   The window-power normalisation factor must be `Nw = Σ w[n]²`, **not**
   `Nw = N`. Using `N` is correct only for a rectangular window; for any
   other window (Hann, Hamming, Blackman …) it under-normalises the PSD,
   raising the apparent noise floor by several dB.

   Correct formula:
   ```
   L(f_k) = |Φ[k]|² / (fs · Nw)   where Nw = Σ w[n]²
   ```

2. **Anti-aliasing filter in decimation (`pn_calc.zip`)**  
   The original code decimated by taking every M-th sample (`data[::M]`)
   without first low-pass filtering. High-frequency noise aliases into the
   analysis band, artificially raising the noise floor.  
   The fix applies a linear-phase FIR low-pass filter at the new Nyquist
   frequency `fs/(2M)` before downsampling (`scipy.signal.decimate` with
   `ftype='fir'`, `zero_phase=True`).

---

### `adev_calc.py` — Allan Deviation (阿伦偏差)

| Function | Description |
|---|---|
| `calc_adev(phase_data, tau0, taus, mode)` | ADEV or overlapping ADEV (OADEV) |
| `calc_mdev(phase_data, tau0, taus)` | Modified Allan deviation (MDEV) |

Implements the standard IEEE 1139 formulas:

```
OAVAR(τ) = 1/(2·(N−2m)·τ²) · Σ_{i=0}^{N−2m−1} (x[i+2m] − 2·x[i+m] + x[i])²
```

The Allan algorithm was already essentially correct in the reference files;
this module is included for completeness and verified with the expected
power-law slopes for white FM noise (ADEV ∝ τ⁻¹/²) and white PM noise
(MDEV ∝ τ⁻³/²).

---

## Quick start

```python
import numpy as np
from pn_calc  import calc_phase_noise, decimate_phase
from adev_calc import calc_adev

# --- phase noise ---
fs   = 1e6            # 1 MHz sample rate
phi  = np.random.randn(2**18)   # synthetic phase fluctuations [rad]
f, L = calc_phase_noise(phi, fs, window='hann')   # L in dBc/Hz

# multi-decade: decimate ×10 each time to reach lower offsets
from pn_calc import calc_phase_noise_multidecade
f_all, L_all = calc_phase_noise_multidecade(phi, fs, decades=3)

# --- Allan deviation ---
tau0 = 1 / fs
taus, adev = calc_adev(phi, tau0, mode='oadev')
```

---

## Running the tests

```bash
pip install numpy scipy pytest
python -m pytest tests/test_pn_adev.py -v
```
