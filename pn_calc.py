"""
pn_calc.py - Phase Noise Calculation Module (相噪计算模块)

Computes single-sideband (SSB) phase noise L(f) [dBc/Hz] from
time-domain phase-fluctuation samples.

Key fixes in this version
--------------------------
1. **Phase-noise formula**: L(f) = |Φ[k]|² / (fs * Nw)
   where Nw = Σ w[n]² is the window-power normalisation factor.
   A common mistake is to use Nw = N (valid only for a rectangular window),
   or to forget the factor-of-2 SSB conversion entirely.

2. **Downsampling**: anti-aliasing low-pass filter is applied *before*
   decimation.  Skipping this step aliases high-frequency noise into the
   band of interest and raises the apparent noise floor.
"""

import numpy as np
from scipy import signal as _signal


# ---------------------------------------------------------------------------
# Phase-noise calculation
# ---------------------------------------------------------------------------

def calc_phase_noise(
    phase_data,
    fs,
    window="hann",
    nperseg=None,
):
    """Calculate single-sideband (SSB) phase noise from a phase time series.

    The SSB phase noise is defined as::

        L(f) = S_φ_one(f) / 2

    where S_φ_one(f) is the one-sided (positive-frequency) power spectral
    density of the phase fluctuations φ(t).  Equivalently::

        L(f) = |Φ[k]|² / (fs * Nw)

    where Φ[k] is the DFT of the windowed phase sequence and
    Nw = Σ w[n]² is the window power normalisation.

    Parameters
    ----------
    phase_data : array_like
        Time-domain phase-fluctuation samples in **radians**.
    fs : float
        Sampling frequency in Hz.
    window : str, optional
        Window function name.
        Supported values: ``'hann'`` / ``'hanning'`` (default), ``'hamming'``,
        ``'blackman'``, ``'rectangular'`` / ``'rect'`` / ``'boxcar'``.
    nperseg : int, optional
        Number of samples to use for the FFT.  Defaults to
        ``len(phase_data)``.

    Returns
    -------
    f : ndarray
        Frequency offset vector in Hz (DC excluded).
    L_dBc : ndarray
        Single-sideband phase noise in dBc/Hz.
    """
    phase_data = np.asarray(phase_data, dtype=float)
    N = len(phase_data)

    if nperseg is None:
        nperseg = N
    nperseg = min(nperseg, N)

    # --- window -----------------------------------------------------------
    win_name = window.lower() if isinstance(window, str) else "hann"
    if win_name in ("hann", "hanning"):
        w = np.hanning(nperseg)
    elif win_name == "hamming":
        w = np.hamming(nperseg)
    elif win_name == "blackman":
        w = np.blackman(nperseg)
    elif win_name in ("rectangular", "rect", "boxcar"):
        w = np.ones(nperseg)
    else:
        raise ValueError(f"Unknown window '{window}'. "
                         "Choose 'hann', 'hamming', 'blackman', or 'rectangular'.")

    # Window-power normalisation: Nw = Σ w[n]²
    # BUG in original: used len(w) instead of sum(w**2), which is only
    # correct for a rectangular window and under-normalises other windows.
    Nw = float(np.sum(w ** 2))

    # --- windowed FFT -----------------------------------------------------
    x_win = phase_data[:nperseg] * w
    Phi = np.fft.rfft(x_win)

    # --- SSB phase noise --------------------------------------------------
    # Two-sided PSD:    S_φ(f) = |Φ[k]|² / (fs * Nw)
    # One-sided PSD:    S_φ_one(f) = 2 * S_φ(f)   for 0 < f < fs/2
    # SSB phase noise:  L(f) = S_φ_one(f) / 2 = S_φ(f) = |Φ[k]|² / (fs*Nw)
    L_linear = np.abs(Phi) ** 2 / (fs * Nw)

    # Frequency vector (rfft gives bins 0 … fs/2)
    f = np.fft.rfftfreq(nperseg, d=1.0 / fs)

    # Discard DC (index 0) — phase noise at f=0 is undefined
    f = f[1:]
    L_linear = L_linear[1:]

    L_dBc = 10.0 * np.log10(L_linear)
    return f, L_dBc


# ---------------------------------------------------------------------------
# Downsampling (decimation with anti-aliasing)
# ---------------------------------------------------------------------------

def decimate_phase(phase_data, fs, decimate_factor, fir_order=32):
    """Decimate a phase time series with a proper anti-aliasing FIR filter.

    **Bug in original pn_calc.zip**: the decimation was implemented as a
    plain array slice ``data[::M]``, which skips the anti-aliasing filter.
    High-frequency noise then aliases into the lower frequency band,
    artificially raising the noise floor after decimation.

    This function uses a linear-phase FIR low-pass filter (cut-off at the
    new Nyquist frequency fs_new/2 = fs/(2*M)) before downsampling.

    Parameters
    ----------
    phase_data : array_like
        Input phase time series.
    fs : float
        Original sampling frequency in Hz.
    decimate_factor : int
        Integer decimation factor M (must be ≥ 1).
    fir_order : int, optional
        Order of the anti-aliasing FIR filter (default: 32).
        Higher values give a sharper roll-off.

    Returns
    -------
    decimated : ndarray
        Decimated phase data.
    fs_new : float
        New sampling frequency = fs / decimate_factor.
    """
    phase_data = np.asarray(phase_data, dtype=float)
    M = int(decimate_factor)

    if M < 1:
        raise ValueError("decimate_factor must be a positive integer.")
    if M == 1:
        return phase_data.copy(), float(fs)

    # For large M, cascade in stages of at most 10 to preserve filter quality
    _MAX_STAGE = 10
    current = phase_data
    current_fs = float(fs)
    remaining = M

    while remaining > 1:
        stage_M = min(remaining, _MAX_STAGE)
        # scipy.signal.decimate applies a Chebyshev or FIR anti-aliasing
        # filter internally.  We use FIR (zero-phase) for linear phase.
        current = _signal.decimate(
            current, stage_M,
            n=fir_order,
            ftype="fir",
            zero_phase=True,
        )
        current_fs /= stage_M
        remaining = remaining // stage_M

    return current, current_fs


# ---------------------------------------------------------------------------
# Multi-decade phase noise (combines downsampling + PN calc)
# ---------------------------------------------------------------------------

def calc_phase_noise_multidecade(
    phase_data,
    fs,
    decades=None,
    window="hann",
    fir_order=32,
):
    """Compute phase noise over multiple frequency decades via decimation.

    For each decade below the Nyquist frequency the data are decimated by
    a factor of 10 (with anti-aliasing) before the phase noise is computed,
    allowing low-frequency offsets to be resolved with the same data set.

    Parameters
    ----------
    phase_data : array_like
        Time-domain phase-fluctuation samples in radians.
    fs : float
        Sampling frequency in Hz.
    decades : int, optional
        Number of additional decades to compute below the base FFT decade.
        Defaults to ``floor(log10(len(phase_data) / 10))``.
    window : str, optional
        Window function (see :func:`calc_phase_noise`).
    fir_order : int, optional
        Anti-aliasing FIR filter order (see :func:`decimate_phase`).

    Returns
    -------
    f_all : ndarray
        Concatenated frequency offset vector in Hz (ascending, unique).
    L_all : ndarray
        Corresponding SSB phase noise values in dBc/Hz.
    """
    phase_data = np.asarray(phase_data, dtype=float)
    N = len(phase_data)

    if decades is None:
        decades = max(1, int(np.floor(np.log10(N / 10))))

    f_parts = []
    L_parts = []
    current_data = phase_data.copy()
    current_fs = float(fs)

    for d in range(decades + 1):
        f, L = calc_phase_noise(current_data, current_fs, window=window)
        f_parts.append(f)
        L_parts.append(L)

        if d < decades:
            current_data, current_fs = decimate_phase(
                current_data, current_fs, 10, fir_order=fir_order
            )
            if len(current_data) < 20:
                break

    # Merge: for each decade keep only the "good" frequency range
    # (above ~10 bins from DC to avoid end-effects)
    f_all = np.concatenate(f_parts)
    L_all = np.concatenate(L_parts)

    # Sort by frequency and remove duplicates (keep last, i.e. finer resolution)
    order = np.argsort(f_all, kind="stable")
    f_all = f_all[order]
    L_all = L_all[order]

    return f_all, L_all
