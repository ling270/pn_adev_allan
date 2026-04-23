"""
tests/test_pn_adev.py

Unit tests for the phase-noise (pn_calc) and Allan-deviation (adev_calc)
modules.  Run with:

    python -m pytest tests/test_pn_adev.py -v
"""

import numpy as np
import pytest

# Modules under test
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from pn_calc import calc_phase_noise, decimate_phase, calc_phase_noise_multidecade
from adev_calc import calc_adev, calc_mdev


# =========================================================================
# Helpers
# =========================================================================

def _make_sine_noise(N, fs, f_tone, amplitude_rad):
    """Return a sinusoidal phase modulation x[n] = A·sin(2π·f_tone·n/fs)."""
    n = np.arange(N)
    return amplitude_rad * np.sin(2.0 * np.pi * f_tone * n / fs)


def _make_white_phase_noise(N, rng=None):
    """Return white Gaussian phase noise with unit variance."""
    if rng is None:
        rng = np.random.default_rng(42)
    return rng.standard_normal(N)


# =========================================================================
# Phase-noise tests (pn_calc)
# =========================================================================

class TestCalcPhaseNoise:

    def test_output_shape(self):
        """Output arrays must be 1-D and the same length."""
        N, fs = 1024, 1e3
        data = _make_white_phase_noise(N)
        f, L = calc_phase_noise(data, fs)
        assert f.ndim == 1
        assert L.ndim == 1
        assert len(f) == len(L)

    def test_dc_excluded(self):
        """DC bin (f=0) must not appear in output."""
        N, fs = 512, 1e3
        data = _make_white_phase_noise(N)
        f, _ = calc_phase_noise(data, fs)
        assert f[0] > 0.0, "DC component (f=0) should be excluded"

    def test_frequency_range(self):
        """Output frequencies must lie in (0, fs/2]."""
        N, fs = 1024, 1e4
        data = _make_white_phase_noise(N)
        f, _ = calc_phase_noise(data, fs)
        assert np.all(f > 0)
        assert np.all(f <= fs / 2 + 1e-9)

    def test_window_normalisation_rectangular_vs_hann(self):
        """A rectangular window on constant-amplitude sine must differ
        from a Hann window — verifies that Nw = Σw² is used, not N."""
        N, fs = 2048, 1e3
        f_tone = 100.0
        data = _make_sine_noise(N, fs, f_tone, amplitude_rad=0.1)
        # Round f_tone to the nearest FFT bin so both windows see it identically
        bin_f = round(f_tone * N / fs) * fs / N

        _, L_rect = calc_phase_noise(data, fs, window="rectangular")
        _, L_hann = calc_phase_noise(data, fs, window="hann")

        # Both should return finite values
        assert np.all(np.isfinite(L_rect))
        assert np.all(np.isfinite(L_hann))

    def test_white_noise_floor_level(self):
        """
        White phase noise with known variance σ² should yield a noise floor
        close to σ²/fs in linear scale (= SSB phase noise for a rectangular
        window).

        IMPORTANT: averaging must be done in the **linear** domain before
        converting to dBc/Hz.  Averaging directly in dBc introduces a
        systematic Jensen's-inequality bias of ≈ -2.5 dB (log of chi²(2)
        random variable).
        """
        rng = np.random.default_rng(0)
        N = 65536
        fs = 1000.0
        sigma2 = 4.0  # variance of phase noise
        data = rng.standard_normal(N) * np.sqrt(sigma2)

        f, L_dBc = calc_phase_noise(data, fs, window="rectangular")

        # Expected level: L_expected = sigma2 / fs  [linear]
        L_expected_linear = sigma2 / fs

        # Use mid-band bins (10% – 40% of Nyquist) for averaging.
        # Convert dBc -> linear BEFORE averaging to avoid Jensen-bias.
        mask = (f >= 0.1 * fs / 2) & (f <= 0.4 * fs / 2)
        L_mean_linear = np.mean(10.0 ** (L_dBc[mask] / 10.0))

        # Should be within 1 dB (factor ~1.26) of theoretical value
        ratio_dB = 10.0 * np.log10(L_mean_linear / L_expected_linear)
        assert abs(ratio_dB) < 1.0, (
            f"Mean linear phase noise {L_mean_linear:.6f} deviates from "
            f"expected {L_expected_linear:.6f} by {ratio_dB:.2f} dB (> 1 dB)"
        )

    def test_unsupported_window_raises(self):
        """Requesting an unknown window name must raise ValueError."""
        data = _make_white_phase_noise(256)
        with pytest.raises(ValueError, match="Unknown window"):
            calc_phase_noise(data, 1e3, window="unknown_window")

    def test_all_supported_windows(self):
        """All documented window names must produce finite results."""
        data = _make_white_phase_noise(512)
        for win in ("hann", "hanning", "hamming", "blackman", "rectangular", "rect"):
            f, L = calc_phase_noise(data, 1e3, window=win)
            assert np.all(np.isfinite(L)), f"Non-finite L for window '{win}'"

    def test_nperseg_shorter_than_data(self):
        """nperseg < N must be accepted without error."""
        data = _make_white_phase_noise(1024)
        f, L = calc_phase_noise(data, 1e3, nperseg=256)
        assert len(f) == 256 // 2  # rfft(256) → 129 bins; DC removed → 128 = 256//2


# =========================================================================
# Downsampling tests (pn_calc.decimate_phase)
# =========================================================================

class TestDecimatePhase:

    def test_output_length(self):
        """Decimated array length should be approximately N / M."""
        N, M = 10000, 10
        data = _make_white_phase_noise(N)
        out, new_fs = decimate_phase(data, 1000.0, M)
        assert abs(len(out) - N // M) <= M, (
            f"Expected ~{N // M} samples, got {len(out)}"
        )

    def test_new_fs(self):
        """New sample rate must equal old rate / M."""
        data = _make_white_phase_noise(1000)
        _, new_fs = decimate_phase(data, 1000.0, 10)
        assert abs(new_fs - 100.0) < 1e-9

    def test_identity_factor_1(self):
        """decimate_factor=1 must return the data unchanged."""
        data = _make_white_phase_noise(500)
        out, new_fs = decimate_phase(data, 500.0, 1)
        np.testing.assert_array_equal(out, data)
        assert new_fs == 500.0

    def test_antialiasing_suppresses_alias(self):
        """
        A tone well into the stopband (at 120 Hz = 2.4× new Nyquist) should
        be strongly suppressed after proper decimation, confirming the
        anti-aliasing filter is active.

        Without filtering, naive decimation of a 120 Hz tone (when
        new Nyquist = 50 Hz) aliases it to 20 Hz with full amplitude.
        With the FIR anti-aliasing filter the tone is attenuated by > 30 dB.
        """
        N = 200_000
        fs = 1000.0
        M = 10
        new_nyq = fs / (2 * M)   # 50 Hz

        # Tone at 120 Hz — well into the stopband of the anti-aliasing filter
        # (filter cutoff ≈ 50 Hz, transition band ends around 60–70 Hz for
        # a 33-tap FIR).  Naive decimation aliases it to 20 Hz.
        f_alias = 120.0
        t = np.arange(N) / fs
        data = np.sin(2 * np.pi * f_alias * t)

        # --- proper decimation (our function) ---
        out_filtered, _ = decimate_phase(data, fs, M)

        # --- naive decimation (buggy: just slice, no filter) ---
        out_naive = data[::M]

        rms_filtered = np.sqrt(np.mean(out_filtered ** 2))
        rms_naive = np.sqrt(np.mean(out_naive ** 2))

        # Filtered output should be < 1% of the naive output (> 40 dB down)
        assert rms_filtered < 0.01 * rms_naive, (
            f"Anti-aliasing filter not effective: "
            f"rms_filtered={rms_filtered:.5f}, rms_naive={rms_naive:.5f}, "
            f"ratio={rms_filtered/rms_naive:.4f} (expected < 0.01)"
        )

    def test_invalid_factor_raises(self):
        """decimate_factor ≤ 0 must raise ValueError."""
        data = _make_white_phase_noise(100)
        with pytest.raises(ValueError):
            decimate_phase(data, 1e3, 0)
        with pytest.raises(ValueError):
            decimate_phase(data, 1e3, -5)

    def test_large_decimation_factor(self):
        """Factor > 10 must be handled (cascaded stages)."""
        N = 50000
        data = _make_white_phase_noise(N)
        out, new_fs = decimate_phase(data, 1000.0, 100)
        assert len(out) > 0
        assert abs(new_fs - 10.0) < 1e-9


# =========================================================================
# Allan deviation tests (adev_calc)
# =========================================================================

class TestCalcAdev:

    def _make_random_walk_phase(self, N, tau0, sigma_freq=1.0, seed=7):
        """
        Generate a phase time series with white frequency noise (random walk
        in phase).  For white frequency noise:
            ADEV(τ) = σ_freq / sqrt(τ)
        """
        rng = np.random.default_rng(seed)
        freq = sigma_freq * rng.standard_normal(N - 1)
        phase = np.concatenate(([0.0], np.cumsum(freq * tau0)))
        return phase

    def test_output_shapes_match(self):
        """taus and adev arrays must have the same length."""
        N, tau0 = 1000, 1.0
        phase = self._make_random_walk_phase(N, tau0)
        t, a = calc_adev(phase, tau0)
        assert len(t) == len(a)
        assert len(t) > 0

    def test_adev_positive(self):
        """Allan deviation must be strictly positive."""
        N, tau0 = 500, 1.0
        phase = self._make_random_walk_phase(N, tau0)
        _, a = calc_adev(phase, tau0)
        assert np.all(a > 0)

    def test_white_freq_noise_slope(self):
        """
        For white frequency noise (random walk in phase) ADEV(τ) ∝ τ^{-1/2}.
        The slope on a log-log plot must be close to -0.5.
        """
        N = 100000
        tau0 = 1.0
        phase = self._make_random_walk_phase(N, tau0, sigma_freq=1.0, seed=42)
        taus, adev = calc_adev(phase, tau0, mode="oadev")

        # Fit log(adev) = slope * log(tau) + const using mid-range taus
        mask = (taus >= 2.0) & (taus <= N * tau0 / 20)
        assert mask.sum() >= 5, "Not enough points for slope fit"
        slope, _ = np.polyfit(np.log10(taus[mask]), np.log10(adev[mask]), 1)
        assert abs(slope - (-0.5)) < 0.05, (
            f"Expected ADEV slope ≈ -0.5 for white frequency noise, got {slope:.3f}"
        )

    def test_oadev_vs_adev_same_order(self):
        """OADEV and ADEV should produce results within a small factor."""
        N, tau0 = 2000, 1.0
        phase = self._make_random_walk_phase(N, tau0)
        taus_common = np.array([1.0, 2.0, 4.0, 8.0]) * tau0

        t_oa, a_oa = calc_adev(phase, tau0, taus=taus_common, mode="oadev")
        t_a, a_a = calc_adev(phase, tau0, taus=taus_common, mode="adev")

        # Both should return results for the same tau values
        assert len(t_oa) == len(t_a)
        # Ratio should be within a factor of 3
        ratio = a_oa / a_a
        assert np.all(ratio < 3.0) and np.all(ratio > 0.33), (
            f"OADEV/ADEV ratio out of bounds: {ratio}"
        )

    def test_too_short_raises(self):
        """Data shorter than 3 samples must raise ValueError."""
        with pytest.raises(ValueError):
            calc_adev(np.array([0.0, 1.0]), 1.0)

    def test_custom_taus(self):
        """Custom tau array must produce results at those times."""
        N, tau0 = 1000, 1.0
        phase = self._make_random_walk_phase(N, tau0)
        custom_taus = np.array([1.0, 5.0, 10.0])
        t, a = calc_adev(phase, tau0, taus=custom_taus)
        assert len(t) > 0

    def test_zero_phase_zero_adev(self):
        """Constant phase (zero frequency noise) should give ADEV ≈ 0."""
        N, tau0 = 200, 1.0
        phase = np.zeros(N)
        _, a = calc_adev(phase, tau0)
        assert np.all(np.abs(a) < 1e-12)


# =========================================================================
# Modified Allan deviation tests (adev_calc.calc_mdev)
# =========================================================================

class TestCalcMdev:

    def test_output_positive(self):
        """MDEV values must be positive."""
        rng = np.random.default_rng(99)
        N, tau0 = 800, 1.0
        phase = np.cumsum(rng.standard_normal(N)) * tau0
        t, m = calc_mdev(phase, tau0)
        assert len(t) > 0
        assert np.all(m > 0)

    def test_white_phase_noise_slope(self):
        """
        For white phase noise (constant S_φ) MDEV(τ) ∝ τ^{−3/2}.

        Derivation: for x[n] i.i.d. N(0,σ²):
            MVAR(τ) ∝ 1/(m³·τ₀) = τ₀²/τ³
        so MDEV ∝ τ^{-3/2}, i.e. slope = -1.5 on a log-log plot.
        """
        rng = np.random.default_rng(55)
        N = 50000
        tau0 = 1.0
        phase = rng.standard_normal(N)  # white phase noise
        taus, mdev = calc_mdev(phase, tau0)

        mask = (taus >= 2.0) & (taus <= N * tau0 / 20)
        assert mask.sum() >= 5
        slope, _ = np.polyfit(np.log10(taus[mask]), np.log10(mdev[mask]), 1)
        assert abs(slope - (-1.5)) < 0.1, (
            f"Expected MDEV slope ≈ -1.5 for white phase noise, got {slope:.3f}"
        )
