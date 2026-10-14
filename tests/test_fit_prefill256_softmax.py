from __future__ import annotations

import numpy as np

from tools.fit_prefill256_softmax import ROOT, _fit, _portable_path, _required_interval


def test_required_interval_widens_only_observed_side() -> None:
    assert _required_interval((0.01, 2.0), (0.02, 1.8), margin=0.2) is None
    assert _required_interval((0.01, 2.0), (0.005, 1.8), margin=0.2) == (
        0.004,
        2.0,
    )
    assert _required_interval((0.01, 2.0), (0.02, 3.0), margin=0.2) == (
        0.01,
        3.5999999999999996,
    )


def test_refit_keeps_at_least_the_original_degree() -> None:
    degree, coefficients, error = _fit(
        np.exp, -3.0, 1.0, initial_degree=17, tol=2.0**-12
    )
    assert degree >= 17
    assert len(coefficients) == degree + 1
    assert error <= 2.0**-12


def test_portable_source_path() -> None:
    assert _portable_path(ROOT / "assets/polynomials/softmax_poly_coeffs.json") == (
        "assets/polynomials/softmax_poly_coeffs.json"
    )
