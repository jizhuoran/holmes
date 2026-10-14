from __future__ import annotations

import numpy as np
import pytest


def test_nonlinear_package_exports_selected_kernels() -> None:
    from llama3fhe.operators.nonlinear import (
        BootstrapConfig,
        BootstrapOperator,
        eval_chebyshev_series_cipher,
        eval_exp_cipher,
        eval_invsqrt_cipher,
        eval_silu_chebyshev_cipher,
    )

    assert callable(eval_chebyshev_series_cipher)
    assert BootstrapConfig.__name__ == "BootstrapConfig"
    assert BootstrapOperator.__name__ == "BootstrapOperator"
    assert callable(eval_exp_cipher)
    assert callable(eval_invsqrt_cipher)
    assert callable(eval_silu_chebyshev_cipher)


def test_static_wrappers_forward_coefficients_without_layout_policy(monkeypatch) -> None:
    from llama3fhe.operators.nonlinear import elementwise

    calls: list[tuple[object, np.ndarray, float, float, object]] = []

    def fake_eval(
        input_cipher,
        coefficients,
        *,
        lower_bound,
        upper_bound,
        crypto_context,
        input_is_chebyshev_mapped=False,
    ):
        assert input_is_chebyshev_mapped is False
        calls.append(
            (
                input_cipher,
                np.asarray(coefficients),
                float(lower_bound),
                float(upper_bound),
                crypto_context,
            )
        )
        return "output"

    monkeypatch.setattr(
        elementwise, "eval_chebyshev_series_cipher", fake_eval
    )

    cipher = object()
    context = object()
    assert elementwise.eval_exp_cipher(
        cipher,
        {"fit_interval": [-2.0, 1.0], "coefficients": [1.0, 2.0]},
        crypto_context=context,
    ) == "output"
    assert elementwise.eval_invsqrt_cipher(
        cipher,
        {"chebyshev_fhe": {"fit_interval": [1.0, 4.0], "coefficients": [4.0]}},
        crypto_context=context,
    ) == "output"
    assert elementwise.eval_silu_chebyshev_cipher(
        cipher,
        [5.0, 6.0],
        lower_bound=-8.0,
        upper_bound=8.0,
        crypto_context=context,
    ) == "output"

    assert [call[2:4] for call in calls] == [
        (-2.0, 1.0),
        (1.0, 4.0),
        (-8.0, 8.0),
    ]
    assert all(call[0] is cipher and call[4] is context for call in calls)
    np.testing.assert_array_equal(calls[0][1], np.asarray([1.0, 2.0]))
    np.testing.assert_array_equal(calls[1][1], np.asarray([4.0]))
    np.testing.assert_array_equal(calls[2][1], np.asarray([5.0, 6.0]))


def test_invsqrt_requires_explicit_fhe_coefficients() -> None:
    from llama3fhe.operators.nonlinear.elementwise import eval_invsqrt_cipher

    try:
        eval_invsqrt_cipher(object(), {}, crypto_context=object())
    except ValueError as exc:
        assert "chebyshev_fhe" in str(exc)
    else:
        raise AssertionError("missing chebyshev_fhe coefficients must be rejected")


def test_bootstrap_config_builds_the_production_and_deep_programs() -> None:
    from llama3fhe.operators.nonlinear import (
        BootstrapConfig,
        BootstrapOperator,
    )

    config = BootstrapConfig(
        log_slots=15,
        level_budget=(4, 4),
        output_levels=23,
    )
    assert config.evict_rotation_cache is False
    assert config.spec().log_slots == 15
    assert config.deep_spec() is None

    main = set(
        BootstrapOperator.required_rotations(
            log_n=16,
            log_slots=15,
            level_budget=(4, 4),
        )
    )
    # The deep (3,3) program's CtoS/StoC union is a superset: key planning
    # must include it whenever --softmax-deep-main-bootstrap is enabled.
    with_deep = set(
        BootstrapOperator.required_rotations(
            log_n=16,
            log_slots=15,
            level_budget=(4, 4),
            deep_level_budget=(3, 3),
        )
    )
    assert len(main) == 78
    assert main < with_deep


def test_prepare_two_pass_input_only_mods_down_excess_headroom(monkeypatch) -> None:
    from llama3fhe.operators.nonlinear import bootstrap

    class State:
        def __init__(self, cur_limbs, scale_degree=2, scaling_factor=7.0):
            self.cur_limbs = int(cur_limbs)
            self.scale_degree = int(scale_degree)
            self.scaling_factor = scaling_factor

        def replace(self, **changes):
            values = {
                "cur_limbs": self.cur_limbs,
                "scale_degree": self.scale_degree,
                "scaling_factor": self.scaling_factor,
            }
            values.update(changes)
            return State(**values)

    class Cipher:
        def __init__(self, limbs):
            self.state = State(limbs)

    aligned_calls = []

    def fake_align(cipher, state, context):
        aligned_calls.append((cipher, state, context))
        result = Cipher(state.cur_limbs)
        result.state = state
        return result

    monkeypatch.setattr(bootstrap.fhe, "align_to", fake_align)

    class Context:
        context = object()

    operator = bootstrap.BootstrapOperator(
        crypto_context=Context(),
        config=bootstrap.BootstrapConfig(
            output_levels=19, iterations=2
        ),
        program=object(),
        setup_seconds=0.0,
    )
    low = Cipher(18)
    assert operator.prepare_two_pass_input(low, iterations=2) is low
    one_pass = Cipher(28)
    assert operator.prepare_two_pass_input(one_pass, iterations=1) is one_pass

    high = Cipher(28)
    prepared = operator.prepare_two_pass_input(high, iterations=2)
    assert prepared is not high
    assert prepared.state.cur_limbs == 18
    assert prepared.state.scale_degree == 1
    assert prepared.state.scaling_factor is None
    assert aligned_calls == [(high, prepared.state, operator.crypto_context.context)]


def test_bootstrap_cache_eviction_uses_configured_rotation_strategy(
    monkeypatch,
) -> None:
    from llama3fhe.operators.nonlinear import bootstrap

    required_call = {}
    cleared = {}

    def fake_required_rotations(**kwargs):
        required_call.update(kwargs)
        return (17, -9)

    class Native:
        logN = 16

        def clear_cuda_rotation_cache(self, *, keep_rotations):
            cleared["keep_rotations"] = tuple(keep_rotations)

    class Context:
        context = Native()

    config = bootstrap.BootstrapConfig(
        level_budget=(4, 4),
        deep_level_budget=(3, 3),
        deep_output_levels=25,
        strategy="normal_bsgs",
        dim1=(8, 16),
        baby_step=(4, 8),
        evict_rotation_cache=True,
    )
    operator = bootstrap.BootstrapOperator(
        crypto_context=Context(),
        config=config,
        program=object(),
        deep_program=object(),
        setup_seconds=0.0,
    )
    monkeypatch.setattr(
        bootstrap.BootstrapOperator,
        "required_rotations",
        staticmethod(fake_required_rotations),
    )

    operator._evict_unneeded_rotation_material()

    assert required_call == {
        "log_n": 16,
        "log_slots": 15,
        "level_budget": (4, 4),
        "deep_level_budget": (3, 3),
        "strategy": "normal_bsgs",
        "dim1": (8, 16),
        "baby_step": (4, 8),
        "sparse_log_slots": None,
    }
    assert cleared["keep_rotations"] == (17, -9)


def test_bootstrap_prepare_releases_both_programs_on_deep_init_failure(
    monkeypatch,
) -> None:
    from llama3fhe.operators.nonlinear import bootstrap

    released: list[object] = []

    class Constants:
        def __init__(self, *, fails: bool) -> None:
            self.fails = bool(fails)

        def set_cache_mode(self, *_args, **_kwargs) -> None:
            if self.fails:
                raise RuntimeError("deep init failed")

    class Program:
        def __init__(self, *, fails: bool) -> None:
            self.constants = Constants(fails=fails)

    programs = [Program(fails=False), Program(fails=True)]
    monkeypatch.setattr(
        bootstrap.bs, "generate", lambda *_args, **_kwargs: programs.pop(0)
    )
    monkeypatch.setattr(
        bootstrap,
        "release_if_supported",
        lambda value: released.append(value),
    )

    class Native:
        pass

    class Context:
        context = Native()
        device = "cpu"

    with pytest.raises(RuntimeError, match="deep init failed"):
        bootstrap.BootstrapOperator.prepare(
            Context(),
            bootstrap.BootstrapConfig(
                deep_level_budget=(3, 3), deep_output_levels=25
            ),
        )

    assert len(released) == 2
    assert all(value is not None for value in released)


def test_bootstrap_release_clears_handles_even_when_deep_release_fails(
    monkeypatch,
) -> None:
    from llama3fhe.operators.nonlinear import bootstrap

    main = object()
    deep = object()
    released: list[object] = []

    def fake_release(value) -> None:
        released.append(value)
        if value is deep:
            raise RuntimeError("deep release failed")

    monkeypatch.setattr(bootstrap, "release_if_supported", fake_release)
    operator = bootstrap.BootstrapOperator(
        crypto_context=object(),
        config=bootstrap.BootstrapConfig(),
        program=main,
        deep_program=deep,
        setup_seconds=0.0,
    )

    with pytest.raises(RuntimeError, match="deep release failed"):
        operator.release()

    assert released == [main, deep]
    assert operator.program is None
    assert operator.deep_program is None


def test_bootstrap_config_accepts_several_sparse_programs() -> None:
    """One sparse-slot program per period: the slim RMSNorm row (log_slots 3 for the DecodeFM
    period 8) and the Decode merged-page Softmax row (log_slots 12, period 4096)."""

    from llama3fhe.operators.nonlinear import BootstrapConfig, BootstrapOperator

    single = BootstrapConfig(log_slots=15, level_budget=(5, 5), output_levels=23, sparse_log_slots=7)
    assert single.sparse_log_slots_tuple == (7,)
    assert single.sparse_slot_periods == (128,)
    assert single.sparse_slots == 128
    assert single.sparse_spec().log_slots == 7
    assert single.sparse_level_budget() == (5, 5)

    multi = BootstrapConfig(
        log_slots=15, level_budget=(5, 5), output_levels=23, sparse_log_slots=(12, 3)
    )
    assert multi.sparse_log_slots_tuple == (12, 3)
    assert multi.sparse_slot_periods == (4096, 8)
    specs = multi.sparse_specs()
    assert [spec.log_slots for spec in specs] == [12, 3]
    # the CtoS/StoC budget is clamped to the sparse slot count
    assert specs[0].level_budget == (5, 5)
    assert specs[1].level_budget == (3, 3)
    assert multi.sparse_level_budget(3) == (3, 3)
    assert multi.sparse_spec().log_slots == 12  # legacy first-program view

    with pytest.raises(ValueError, match="distinct"):
        BootstrapConfig(log_slots=15, level_budget=(5, 5), output_levels=23, sparse_log_slots=(7, 7))
    with pytest.raises(ValueError, match="sparse_log_slots must lie"):
        BootstrapConfig(log_slots=15, level_budget=(5, 5), output_levels=23, sparse_log_slots=(7, 15))

    dense = set(BootstrapOperator.required_rotations(log_n=16, log_slots=15, level_budget=(5, 5)))
    one = set(BootstrapOperator.required_rotations(log_n=16, log_slots=15, level_budget=(5, 5), sparse_log_slots=12))
    both = set(
        BootstrapOperator.required_rotations(log_n=16, log_slots=15, level_budget=(5, 5), sparse_log_slots=(12, 3))
    )
    assert dense <= one <= both


def test_bootstrap_operator_sparse_program_table() -> None:
    from llama3fhe.operators.nonlinear import BootstrapConfig, BootstrapOperator

    config = BootstrapConfig(log_slots=15, level_budget=(5, 5), output_levels=23, sparse_log_slots=(12, 3))
    programs = {4096: object(), 8: object()}
    operator = BootstrapOperator(
        crypto_context=None, config=config, program=object(), setup_seconds=0.0, sparse_programs=dict(programs)
    )
    assert operator.sparse_slot_periods == (8, 4096)
    assert operator.supports_periodic_slots(4096) and operator.supports_periodic_slots(8)
    assert not operator.supports_periodic_slots(128)
    assert operator.sparse_program is programs[4096]  # insertion order = config order
    assert operator.sparse_slots == 8  # sorted view

    empty = BootstrapOperator(crypto_context=None, config=config, program=object(), setup_seconds=0.0)
    assert empty.sparse_programs == {}
    assert empty.sparse_program is None and empty.sparse_slots is None
    assert not empty.supports_periodic_slots(8)
