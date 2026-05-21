"""
adev_calc.py - Allan Deviation Calculation Module (阿伦偏差计算模块)

Computes Allan deviation (ADEV) and overlapping Allan deviation (OADEV)
from a phase time series.

Standard references
-------------------
* IEEE Std 1139-2008, "IEEE Standard Definitions of Physical Quantities for
  Fundamental Frequency and Time Metrology — Random Instabilities"
* W. J. Riley, "Handbook of Frequency Analysis", NIST, 2008
"""

import numpy as np


def calc_adev(
    phase_data,
    tau0,
    taus=None,
    mode="oadev",
):
    """Calculate Allan deviation from a phase time series.

    For averaging time τ = m·τ₀ the (overlapping) Allan variance is::

        OAVAR(τ) = 1 / (2·(N−2m)·τ²)
                   · Σ_{i=0}^{N−2m−1}  (x[i+2m] − 2·x[i+m] + x[i])²

    The non-overlapping version steps the index by *m* instead of 1::

        AVAR(τ) = 1 / (2·M·τ²)
                  · Σ_{j=0}^{M−1}  (x[(2j+2)m] − 2·x[(2j+1)m] + x[2j·m])²

    where M = ⌊(N−1)/(2m)⌋.

    Both formulas require at least three phase samples separated by *m*
    samples, i.e. N ≥ 2m + 1.

    Parameters
    ----------
    phase_data : array_like
        Phase time series (radians or seconds — unit consistent with τ₀).
    tau0 : float
        Basic sample interval in seconds.
    taus : array_like, optional
        Averaging times (seconds) at which ADEV is computed.  Defaults to
        ``floor(log-spaced)`` values from τ₀ to N·τ₀/4.
    mode : {'oadev', 'adev'}, optional
        * ``'oadev'`` — overlapping Allan deviation (default, more efficient).
        * ``'adev'``  — standard (non-overlapping) Allan deviation.

    Returns
    -------
    taus_out : ndarray
        Averaging times in seconds (only valid entries returned).
    adev_out : ndarray
        Allan deviation values.
    """
    phase_data = np.asarray(phase_data, dtype=float)
    N = len(phase_data)

    if N < 3:
        raise ValueError("phase_data must have at least 3 samples.")

    # --- build averaging-factor array m -----------------------------------
    if taus is None:
        max_m = max(1, N // 4)
        # Log-spaced, deduplicated integer m values
        ms = np.unique(
            np.floor(np.logspace(0, np.log10(max_m), 200)).astype(int)
        )
    else:
        taus = np.asarray(taus, dtype=float)
        ms = np.round(taus / tau0).astype(int)

    # Keep only m values where we have enough data (N >= 2m + 1)
    ms = ms[(ms >= 1) & (ms <= (N - 1) // 2)]
    if len(ms) == 0:
        raise ValueError(
            "No valid averaging factors: data set is too short for the "
            "requested tau values."
        )

    # --- compute ADEV for each m ------------------------------------------
    adev_out = np.empty(len(ms), dtype=float)

    for idx, m in enumerate(ms):
        m = int(m)
        tau = m * tau0
        x = phase_data

        if mode == "oadev":
            # Overlapping: all N−2m triplets
            n = N - 2 * m
            if n < 1:
                adev_out[idx] = np.nan
                continue
            s = np.sum((x[2 * m:] - 2.0 * x[m:-m] + x[:N - 2 * m]) ** 2)
            adev_out[idx] = np.sqrt(s / (2.0 * n * tau ** 2))

        else:  # 'adev' — non-overlapping
            # Number of non-overlapping triplets: M = floor((N-1) / (2m))
            big_M = (N - 1) // (2 * m)
            if big_M < 1:
                adev_out[idx] = np.nan
                continue
            j = np.arange(big_M)
            s = np.sum(
                (x[(2 * j + 2) * m] - 2.0 * x[(2 * j + 1) * m] + x[2 * j * m]) ** 2
            )
            adev_out[idx] = np.sqrt(s / (2.0 * big_M * tau ** 2))

    taus_out = ms.astype(float) * tau0

    # Remove any NaN entries
    valid = ~np.isnan(adev_out)
    return taus_out[valid], adev_out[valid]


def calc_mdev(phase_data, tau0, taus=None):
    """Calculate Modified Allan deviation (MDEV) from a phase time series.

    The modified Allan variance is::

        MVAR(τ) = 1 / (2·τ²·N·m²)
                  · Σ_{j=0}^{N−3m}  [ Σ_{i=j}^{j+m−1} (x[i+2m]−2·x[i+m]+x[i]) ]²

    Parameters
    ----------
    phase_data : array_like
        Phase time series.
    tau0 : float
        Basic sample interval in seconds.
    taus : array_like, optional
        Averaging times (seconds).

    Returns
    -------
    taus_out : ndarray
        Averaging times in seconds.
    mdev_out : ndarray
        Modified Allan deviation values.
    """
    phase_data = np.asarray(phase_data, dtype=float)
    N = len(phase_data)

    if N < 3:
        raise ValueError("phase_data must have at least 3 samples.")

    if taus is None:
        max_m = max(1, N // 4)
        ms = np.unique(
            np.floor(np.logspace(0, np.log10(max_m), 200)).astype(int)
        )
    else:
        taus = np.asarray(taus, dtype=float)
        ms = np.round(taus / tau0).astype(int)

    ms = ms[(ms >= 1) & (ms <= (N - 1) // 3)]
    if len(ms) == 0:
        raise ValueError(
            "No valid averaging factors for the given data length and taus."
        )

    mdev_out = np.empty(len(ms), dtype=float)

    for idx, m in enumerate(ms):
        m = int(m)
        tau = m * tau0
        x = phase_data

        # Second differences
        d2 = x[2 * m:] - 2.0 * x[m: N - m] + x[:N - 2 * m]
        # d2 has length N-2m

        n_outer = N - 3 * m + 1
        if n_outer < 1:
            mdev_out[idx] = np.nan
            continue

        # Sliding sum over m consecutive second-differences
        # Efficient via cumulative sum
        cs = np.cumsum(np.concatenate(([0.0], d2)))
        inner_sums = cs[m:n_outer + m] - cs[:n_outer]

        s = np.sum(inner_sums ** 2)
        mdev_out[idx] = np.sqrt(s / (2.0 * tau ** 2 * n_outer * m ** 2))

    taus_out = ms.astype(float) * tau0
    valid = ~np.isnan(mdev_out)
    return taus_out[valid], mdev_out[valid]
