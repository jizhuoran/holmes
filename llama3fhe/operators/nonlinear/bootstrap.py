from __future__ import annotations

"""Explicit EasyFHE bootstrap operator.

Bootstrap is a nonlinear HE operation, not model orchestration.  This module
owns the public EasyFHE bootstrap program and the optional two-pass Meta-BTS
correction.  Callers supply the native CKKS context through the project's
execution session; no private EasyFHE kernels or legacy-wheel patches live
here.
"""

import contextlib
import contextvars
import time
from dataclasses import dataclass, field

import easyfhe
import easyfhe.bs.openfhe as bs
from easyfhe import fhe

from llama3fhe.backend import release_if_supported, stage_sync_enabled


#: Call-site label of the bootstrap currently running (``None`` outside any
#: :func:`bootstrap_tag` block).  Pure bookkeeping for the bench timers
#: (``bench/t2_decode.py`` ``bootstrap_records``); never changes the result.
_BOOTSTRAP_TAG: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "llama3fhe_bootstrap_tag", default=None
)


def current_bootstrap_tag() -> str | None:
    return _BOOTSTRAP_TAG.get()


@contextlib.contextmanager
def bootstrap_tag(tag: str):
    """Label every ``BootstrapOperator.refresh`` issued inside the block."""

    token = _BOOTSTRAP_TAG.set(str(tag))
    try:
        yield
    finally:
        _BOOTSTRAP_TAG.reset(token)


@dataclass(frozen=True)
class BootstrapConfig:
    log_slots: int = 15
    level_budget: tuple[int, int] = (4, 4)
    output_levels: int = 24
    iterations: int = 2
    precision_bits: int = 8
    mode: str = "modraise_first"
    # FFT/automorphism strategy.  ``double_hoist`` is fastest in the usual
    # regime; ``normal_bsgs`` keeps fewer extended batches live and is useful
    # when a large-ring bootstrap is memory-bound on 24-GiB cards.
    strategy: str = "double_hoist"
    dim1: tuple[int, int] | None = None
    baby_step: tuple[int, int] | None = None
    # Rotation keys are model/session constants. Keep their CUDA material
    # resident across bootstraps and transformer layers by default; callers
    # may still opt into host-backed eviction for a constrained device.
    evict_rotation_cache: bool = False
    # Optional second full-slot program with a shallower CtoS/StoC budget.
    # Its program depth is smaller, so it may emit more output levels than
    # the main program (e.g. (3,3) -> 25 vs (4,4) -> 23 at depth 41).  Used
    # by refresh sites whose downstream chain is exactly a couple of levels
    # short (the PairFM paired-input threshold).
    deep_level_budget: tuple[int, int] | None = None
    deep_output_levels: int | None = None
    sparse_log_slots: int | tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if (self.deep_level_budget is None) != (self.deep_output_levels is None):
            raise ValueError(
                "deep_level_budget and deep_output_levels must be set together."
            )
        ks = self.sparse_log_slots_tuple
        if len(set(ks)) != len(ks):
            raise ValueError(f"sparse_log_slots must be distinct, got {self.sparse_log_slots}.")
        for k in ks:
            if k <= 0 or k >= int(self.log_slots):
                raise ValueError(
                    "sparse_log_slots must lie in (0, log_slots), got "
                    f"{self.sparse_log_slots} with log_slots={self.log_slots}."
                )

    @property
    def sparse_log_slots_tuple(self) -> tuple[int, ...]:
        """``sparse_log_slots`` normalised to a (possibly empty) tuple of ints."""

        if self.sparse_log_slots is None:
            return ()
        if isinstance(self.sparse_log_slots, int):
            return (int(self.sparse_log_slots),)
        return tuple(int(k) for k in self.sparse_log_slots)

    @property
    def sparse_slot_periods(self) -> tuple[int, ...]:
        """Message periods (in slots) the sparse programs accept."""

        return tuple(1 << k for k in self.sparse_log_slots_tuple)

    @property
    def sparse_slots(self) -> int | None:
        """Period (slots) of the *first* sparse program, or ``None`` (legacy single-program view)."""

        periods = self.sparse_slot_periods
        return periods[0] if periods else None

    def sparse_level_budget(self, log_slots: int | None = None) -> tuple[int, int] | None:
        ks = self.sparse_log_slots_tuple
        if not ks:
            return None
        k = int(ks[0] if log_slots is None else log_slots)
        return tuple(min(int(v), k) for v in self.level_budget)

    def spec(self) -> bs.BootstrapSpec:
        return bs.BootstrapSpec(
            log_slots=int(self.log_slots),
            level_budget=tuple(int(value) for value in self.level_budget),
            output_levels=int(self.output_levels),
            strategy=str(self.strategy),
            mode=str(self.mode),
            dim1=self.dim1,
            baby_step=self.baby_step,
        )

    def sparse_specs(self) -> tuple[bs.BootstrapSpec, ...]:
        """One sparse-slot spec per entry of ``sparse_log_slots`` (same order)."""

        return tuple(
            bs.BootstrapSpec(
                log_slots=int(k),
                level_budget=self.sparse_level_budget(k),
                output_levels=int(self.output_levels),
                strategy=str(self.strategy),
                mode=str(self.mode),
            )
            for k in self.sparse_log_slots_tuple
        )

    def sparse_spec(self) -> bs.BootstrapSpec | None:
        specs = self.sparse_specs()
        return specs[0] if specs else None

    def deep_spec(self) -> bs.BootstrapSpec | None:
        if self.deep_level_budget is None:
            return None
        return bs.BootstrapSpec(
            log_slots=int(self.log_slots),
            level_budget=tuple(int(v) for v in self.deep_level_budget),
            output_levels=int(self.deep_output_levels),
            strategy=str(self.strategy),
            mode=str(self.mode),
            dim1=self.dim1,
            baby_step=self.baby_step,
        )


@dataclass
class BootstrapOperator:
    """Prepared bootstrap program plus observable execution policy."""

    crypto_context: object
    config: BootstrapConfig
    program: object
    setup_seconds: float
    deep_program: object | None = None
    #: ``{period_slots: program}`` -- one sparse-slot program per entry of
    #: ``config.sparse_log_slots`` (empty without any).
    sparse_programs: dict[int, object] = field(default_factory=dict)
    calls: int = 0
    native_calls: int = 0
    sparse_calls: int = 0
    cache_eviction_seconds: float = 0.0
    #: Batched refreshes served by :attr:`batch_backend` (each counts the
    #: ciphertexts it refreshed in ``calls`` as well).
    batch_calls: int = 0
    batch_ciphers: int = 0

    #: Optional batched bootstrap backend, ``callable(ciphers, crypto_context,
    #: program) -> list`` refreshing several ciphertexts of one state through
    #: the dense program in one pass (C2S / EvalMod / S2C on a cipher batch).
    #: ``None`` (default) makes :meth:`refresh_many` loop over :meth:`refresh`.
    #: Installed by ``bench/cmp/holmes_batch_bs.py`` (``HOLMES_BATCH_BS=1``).
    batch_backend = None

    @property
    def sparse_program(self) -> object | None:
        """The first sparse program (legacy single-program view), or ``None``."""

        for program in self.sparse_programs.values():
            return program
        return None

    @property
    def sparse_slot_periods(self) -> tuple[int, ...]:
        """Periods (slots) :meth:`refresh` accepts as ``periodic_slots``."""

        return tuple(sorted(int(p) for p in self.sparse_programs))

    @property
    def sparse_slots(self) -> int | None:
        """Period (slots) of the first sparse program, or ``None``."""

        periods = self.sparse_slot_periods
        return periods[0] if periods else None

    def supports_periodic_slots(self, period: int) -> bool:
        """Whether a sparse program for messages of ``period`` slots exists."""

        return int(period) in self.sparse_programs

    def prepare_two_pass_input(
        self,
        cipher,
        *,
        iterations: int | None = None,
        use_deep_program: bool = False,
    ):
        """Return a score checkpoint input with headroom for Meta-BTS correction.

        The caller owns a new ciphertext when the return value differs from
        ``cipher``.  This is opt-in: ``refresh`` keeps its existing policy.
        """

        selected = self.config.iterations if iterations is None else int(iterations)
        if selected != 2:
            return cipher
        output_limbs = (
            self.config.deep_output_levels
            if use_deep_program
            else self.config.output_levels
        )
        if output_limbs is None or int(output_limbs) <= 2:
            raise ValueError("two-pass input needs bootstrap output_levels > 2")
        target_limbs = int(output_limbs) - 1
        if int(cipher.state.cur_limbs) <= target_limbs:
            return cipher
        target_state = cipher.state.replace(
            cur_limbs=target_limbs,
            scale_degree=1,
            scaling_factor=None,
        )
        return fhe.align_to(cipher, target_state, self.crypto_context.context)

    @classmethod
    def prepare(
        cls,
        crypto_context,
        config: BootstrapConfig,
    ) -> "BootstrapOperator":
        start = time.perf_counter()
        program = None
        deep_program = None
        sparse_programs: dict[int, object] = {}
        try:
            program = bs.generate(crypto_context.context, config.spec())
            program.constants.set_cache_mode("plain", clear=True)
            deep_spec = config.deep_spec()
            deep_program = (
                None
                if deep_spec is None
                else bs.generate(crypto_context.context, deep_spec)
            )
            if deep_program is not None:
                deep_program.constants.set_cache_mode("plain", clear=True)
            for spec in config.sparse_specs():
                sparse_program = bs.generate(crypto_context.context, spec)
                sparse_program.constants.set_cache_mode("plain", clear=True)
                sparse_programs[1 << int(spec.log_slots)] = sparse_program
            if str(crypto_context.device).startswith("cuda"):
                easyfhe.cuda.synchronize()
            return cls(
                crypto_context=crypto_context,
                config=config,
                program=program,
                deep_program=deep_program,
                sparse_programs=sparse_programs,
                setup_seconds=float(time.perf_counter() - start),
            )
        except Exception:
            try:
                release_if_supported(program)
            finally:
                try:
                    if deep_program is not None:
                        release_if_supported(deep_program)
                finally:
                    for sparse_program in sparse_programs.values():
                        release_if_supported(sparse_program)
            raise

    @staticmethod
    def required_rotations(
        *,
        log_n: int,
        log_slots: int = 15,
        level_budget: tuple[int, int] = (4, 4),
        deep_level_budget: tuple[int, int] | None = None,
        strategy: str = "double_hoist",
        dim1: tuple[int, int] | None = None,
        baby_step: tuple[int, int] | None = None,
        sparse_log_slots: int | tuple[int, ...] | None = None,
    ) -> tuple[int, ...]:
        specs = [
            bs.BootstrapSpec(
                log_slots=int(log_slots),
                level_budget=tuple(int(value) for value in level_budget),
                output_levels=0,
                strategy=str(strategy),
                dim1=dim1,
                baby_step=baby_step,
            )
        ]
        if deep_level_budget is not None:
            specs.append(
                bs.BootstrapSpec(
                    log_slots=int(log_slots),
                    level_budget=tuple(
                        int(value) for value in deep_level_budget
                    ),
                    output_levels=0,
                    strategy=str(strategy),
                    dim1=dim1,
                    baby_step=baby_step,
                )
            )
        sparse_ks = (
            ()
            if sparse_log_slots is None
            else ((int(sparse_log_slots),) if isinstance(sparse_log_slots, int) else tuple(int(k) for k in sparse_log_slots))
        )
        for k in sparse_ks:
            specs.append(
                bs.BootstrapSpec(
                    log_slots=k,
                    level_budget=tuple(
                        min(int(value), k) for value in level_budget
                    ),
                    output_levels=0,
                    strategy=str(strategy),
                )
            )
        return tuple(
            bs.requirements(tuple(specs), log_n=int(log_n)).rotations
        )

    def _evict_unneeded_rotation_material(self) -> None:
        if not bool(self.config.evict_rotation_cache):
            return
        clear = getattr(
            self.crypto_context.context,
            "clear_cuda_rotation_cache",
            None,
        )
        if not callable(clear):
            raise RuntimeError(
                "EasyFHE Context.clear_cuda_rotation_cache is required. "
                "Install an EasyFHE build that supports rotation-cache eviction."
            )
        start = time.perf_counter()
        clear(
            keep_rotations=self.required_rotations(
                log_n=int(self.crypto_context.context.logN),
                log_slots=int(self.config.log_slots),
                level_budget=self.config.level_budget,
                deep_level_budget=self.config.deep_level_budget,
                strategy=str(self.config.strategy),
                dim1=self.config.dim1,
                baby_step=self.config.baby_step,
                sparse_log_slots=self.config.sparse_log_slots,
            )
        )
        self.cache_eviction_seconds += time.perf_counter() - start

    def refresh(
        self,
        cipher,
        *,
        iterations: int | None = None,
        precision_bits: int | None = None,
        match_two_pass_level: bool = False,
        use_deep_program: bool = False,
        synchronize: bool = True,
        periodic_slots: int | None = None,
    ):
        """Return a refreshed ciphertext without consuming ``cipher``.

        ``periodic_slots`` asserts that the *message* of ``cipher`` is
        periodic with that period (a power of two with a matching entry in
        ``config.sparse_log_slots``); the ciphertext is then bootstrapped
        through that sparse program as a metadata-only slot view and the
        result is viewed back at the full slot count.  The caller owns that periodicity
        promise: a non-periodic message through the sparse view decodes to
        garbage (the probe's negative control).  Without a matching sparse
        program the request is an error, not a silent dense fallback.
        """

        self._evict_unneeded_rotation_material()
        program = self.program
        if bool(use_deep_program):
            if self.deep_program is None:
                raise ValueError(
                    "deep bootstrap requested without deep_level_budget in "
                    "BootstrapConfig."
                )
            program = self.deep_program
        native_passes = [0]
        selected_iterations = (
            int(self.config.iterations)
            if iterations is None
            else int(iterations)
        )
        full_slots = int(getattr(cipher, "slots", 0))
        source = cipher
        if periodic_slots is not None:
            periodic_slots = int(periodic_slots)
            if bool(use_deep_program):
                raise ValueError("periodic_slots and use_deep_program are exclusive.")
            program = self.sparse_programs.get(periodic_slots)
            if program is None:
                raise ValueError(
                    f"periodic_slots={periodic_slots} needs BootstrapConfig.sparse_log_slots to include "
                    f"{periodic_slots.bit_length() - 1}, operator has {self.config.sparse_log_slots}."
                )
            if full_slots % periodic_slots:
                raise ValueError(
                    f"cipher.slots={full_slots} is not a multiple of periodic_slots={periodic_slots}."
                )
            if full_slots != periodic_slots:
                source = cipher.cipher_like(cipher.cv, slots=periodic_slots)
        result = _bootstrap_with_optional_correction(
            source,
            crypto_context=self.crypto_context.context,
            program=program,
            iterations=selected_iterations,
            precision_bits=(
                int(self.config.precision_bits)
                if precision_bits is None
                else int(precision_bits)
            ),
            native_passes=native_passes,
        )
        if periodic_slots is not None:
            self.sparse_calls += 1
            if int(getattr(result, "slots", full_slots)) != full_slots:
                # ``cipher_like`` shares ``cv``; drop the intermediate view object only.
                result = result.cipher_like(result.cv, slots=full_slots)
        if bool(match_two_pass_level) and selected_iterations == 1:
            target_limbs = int(result.state.cur_limbs) - 1
            if target_limbs <= 1:
                release_if_supported(result)
                raise ValueError(
                    "matching the two-pass bootstrap level leaves too few limbs."
                )
            target_state = result.state.replace(
                cur_limbs=target_limbs,
                scale_degree=1,
                scaling_factor=None,
            )
            try:
                matched = fhe.align_to(
                    result, target_state, self.crypto_context.context
                )
            except Exception:
                release_if_supported(result)
                raise
            if matched is not result:
                release_if_supported(result)
            result = matched
        if (
            bool(synchronize)
            and stage_sync_enabled()
            and str(self.crypto_context.device).startswith("cuda")
        ):
            easyfhe.cuda.synchronize()
        self.calls += 1
        self.native_calls += int(native_passes[0])
        return result

    def refresh_many(
        self,
        ciphers,
        *,
        iterations: int | None = None,
        precision_bits: int | None = None,
        match_two_pass_level: bool = False,
        use_deep_program: bool = False,
        synchronize: bool = True,
    ):
        """``[refresh(c, ...) for c in ciphers]``, batched when a :attr:`batch_backend` is installed.

        The batch path is taken only for the plain dense one-pass program and
        when every input shares ``state`` and ``slots`` (the stations' pair
        bootstrap and the Softmax pre-exp refresh satisfy both); everything
        else falls back to the one-at-a-time loop, so results and bookkeeping
        (``calls``, ``native_calls``) are identical either way.
        """

        ciphers = list(ciphers)
        if not ciphers:
            return []
        selected_iterations = (
            int(self.config.iterations) if iterations is None else int(iterations)
        )
        backend = self.batch_backend
        first = ciphers[0]
        batchable = (
            backend is not None
            and len(ciphers) >= 2
            and selected_iterations in (1, 2)
            and not bool(use_deep_program)
            and all(
                c.state == first.state
                and int(getattr(c, "slots", 0)) == int(getattr(first, "slots", 0))
                and int(getattr(c, "batch_size", 1)) == 1
                for c in ciphers
            )
        )
        if not batchable:
            return [
                self.refresh(
                    c,
                    iterations=iterations,
                    precision_bits=precision_bits,
                    match_two_pass_level=match_two_pass_level,
                    use_deep_program=use_deep_program,
                    synchronize=synchronize,
                )
                for c in ciphers
            ]
        self._evict_unneeded_rotation_material()
        program = self.program
        if int(first.state.cur_limbs) > int(program.raise_to_limbs):
            raise ValueError(
                "bootstrap input has more limbs than the program raise target; "
                f"input cur_limbs={first.state.cur_limbs}, raise_to_limbs={program.raise_to_limbs}"
            )
        native_passes = [0]
        results = _bootstrap_many_with_optional_correction(
            ciphers,
            crypto_context=self.crypto_context.context,
            program=program,
            iterations=selected_iterations,
            precision_bits=(
                int(self.config.precision_bits)
                if precision_bits is None
                else int(precision_bits)
            ),
            native_passes=native_passes,
            bootstrap_many=lambda values: list(
                backend(values, self.crypto_context.context, program)
            ),
        )
        if len(results) != len(ciphers):
            raise RuntimeError("batch bootstrap backend returned a different number of ciphertexts")
        if bool(match_two_pass_level) and selected_iterations == 1:
            matched: list[object] = []
            for result in results:
                target_limbs = int(result.state.cur_limbs) - 1
                if target_limbs <= 1:
                    release_if_supported(result)
                    raise ValueError("matching the two-pass bootstrap level leaves too few limbs.")
                target_state = result.state.replace(
                    cur_limbs=target_limbs, scale_degree=1, scaling_factor=None
                )
                aligned = fhe.align_to(result, target_state, self.crypto_context.context)
                if aligned is not result:
                    release_if_supported(result)
                matched.append(aligned)
            results = matched
        if (
            bool(synchronize)
            and stage_sync_enabled()
            and str(self.crypto_context.device).startswith("cuda")
        ):
            easyfhe.cuda.synchronize()
        self.calls += len(results)
        self.native_calls += int(native_passes[0])
        self.batch_calls += 1 if selected_iterations == 1 else 2
        self.batch_ciphers += len(results)
        return results

    def release(self) -> None:
        try:
            release_if_supported(self.program)
        finally:
            try:
                if self.deep_program is not None:
                    release_if_supported(self.deep_program)
            finally:
                try:
                    for sparse_program in self.sparse_programs.values():
                        release_if_supported(sparse_program)
                finally:
                    self.program = None
                    self.deep_program = None
                    self.sparse_programs = {}


def _multiply_integer(cipher, value: int, *, crypto_context):
    encoded = fhe.encode_scalar(
        int(value),
        cur_limbs=int(cipher.state.cur_limbs),
        scale_degree=0,
        mode="integer",
        context=crypto_context,
    )
    return fhe.homo_mul_scalar(cipher, encoded, crypto_context)


def _multiply_double_rescale(cipher, value: float, *, crypto_context):
    encoded = fhe.encode_scalar(
        float(value),
        cur_limbs=int(cipher.state.cur_limbs),
        scale_degree=1,
        scaling_factor=cipher.state.scaling_factor,
        context=crypto_context,
    )
    return fhe.homo_mul_scalar_rescale(cipher, encoded, crypto_context)


def _bootstrap_with_optional_correction(
    cipher,
    *,
    crypto_context,
    program,
    iterations: int,
    precision_bits: int,
    native_passes: list[int] | None = None,
):
    """Apply one bootstrap or the existing two-pass Meta-BTS correction."""

    iterations = int(iterations)
    if iterations not in (1, 2):
        raise ValueError("bootstrap iterations must be one or two.")

    def once(value):
        if native_passes is not None:
            native_passes[0] += 1
        return bs.bootstrap(value, crypto_context, program)

    if iterations == 1:
        return once(cipher)

    initial_limbs = int(cipher.state.cur_limbs)
    original = cipher.deep_copy()
    first = once(cipher)
    first_limbs = int(first.state.cur_limbs)
    if first_limbs <= initial_limbs:
        release_if_supported(original)
        return first

    try:
        factor = 1 << int(precision_bits)
        if int(precision_bits) <= 0:
            first_down = fhe.align_to(
                first, original.state, crypto_context
            )
            original_aligned = fhe.align_to(
                original, first_down.state, crypto_context
            )
            error_input = fhe.homo_sub(
                first_down, original_aligned, crypto_context
            )
            error = once(error_input)
            first_aligned = fhe.align_to(
                first, error.state, crypto_context
            )
            error_aligned = fhe.align_to(
                error, first_aligned.state, crypto_context
            )
            return fhe.homo_sub(first_aligned, error_aligned, crypto_context)

        first_scaled = _multiply_integer(
            first, factor, crypto_context=crypto_context
        )
        first_down = fhe.align_to(
            first_scaled, original.state, crypto_context
        )
        original_scaled = _multiply_integer(
            original, factor, crypto_context=crypto_context
        )
        original_scaled = fhe.align_to(
            original_scaled, first_down.state, crypto_context
        )
        error_input = fhe.homo_sub(
            first_down, original_scaled, crypto_context
        )
        error = once(error_input)
        first_aligned = fhe.align_to(
            first_scaled, error.state, crypto_context
        )
        error_aligned = fhe.align_to(
            error, first_aligned.state, crypto_context
        )
        corrected = fhe.homo_sub(
            first_aligned, error_aligned, crypto_context
        )
        return _multiply_double_rescale(
            corrected,
            1.0 / float(factor),
            crypto_context=crypto_context,
        )
    except ValueError:
        return first
    finally:
        release_if_supported(original)


def _bootstrap_many_with_optional_correction(
    ciphers,
    *,
    crypto_context,
    program,
    iterations: int,
    precision_bits: int,
    native_passes: list[int] | None,
    bootstrap_many,
):
    """:func:`_bootstrap_with_optional_correction` over a list, with every bootstrap pass batched.

    ``bootstrap_many(values)`` bootstraps a list of same-state ciphertexts in
    one call (``BootstrapOperator.batch_backend``); the Meta-BTS correction
    then runs its two passes as two batched calls instead of two per
    ciphertext.  The per-ciphertext arithmetic is the single-cipher routine's,
    step for step (same result, same ``native_passes`` count).
    """

    iterations = int(iterations)
    if iterations not in (1, 2):
        raise ValueError("bootstrap iterations must be one or two.")
    ciphers = list(ciphers)

    def many(values):
        if native_passes is not None:
            native_passes[0] += len(values)
        return list(bootstrap_many(values))

    if iterations == 1:
        return many(ciphers)

    initial_limbs = [int(c.state.cur_limbs) for c in ciphers]
    originals = [c.deep_copy() for c in ciphers]
    try:
        firsts = many(ciphers)
        if all(
            int(first.state.cur_limbs) <= limbs
            for first, limbs in zip(firsts, initial_limbs)
        ):
            return firsts
        factor = 1 << int(precision_bits)
        error_inputs = []
        scaled_firsts = []
        for first, original in zip(firsts, originals):
            if int(precision_bits) <= 0:
                first_down = fhe.align_to(first, original.state, crypto_context)
                original_aligned = fhe.align_to(original, first_down.state, crypto_context)
                error_inputs.append(fhe.homo_sub(first_down, original_aligned, crypto_context))
                scaled_firsts.append(first)
            else:
                first_scaled = _multiply_integer(first, factor, crypto_context=crypto_context)
                first_down = fhe.align_to(first_scaled, original.state, crypto_context)
                original_scaled = _multiply_integer(original, factor, crypto_context=crypto_context)
                original_scaled = fhe.align_to(original_scaled, first_down.state, crypto_context)
                error_inputs.append(fhe.homo_sub(first_down, original_scaled, crypto_context))
                scaled_firsts.append(first_scaled)
        try:
            errors = many(error_inputs)
            out = []
            for first_scaled, error in zip(scaled_firsts, errors):
                first_aligned = fhe.align_to(first_scaled, error.state, crypto_context)
                error_aligned = fhe.align_to(error, first_aligned.state, crypto_context)
                corrected = fhe.homo_sub(first_aligned, error_aligned, crypto_context)
                if int(precision_bits) <= 0:
                    out.append(corrected)
                else:
                    out.append(
                        _multiply_double_rescale(
                            corrected, 1.0 / float(factor), crypto_context=crypto_context
                        )
                    )
            return out
        except ValueError:
            return firsts
    finally:
        for original in originals:
            release_if_supported(original)


__all__ = ["BootstrapConfig", "BootstrapOperator", "bootstrap_tag", "current_bootstrap_tag"]
