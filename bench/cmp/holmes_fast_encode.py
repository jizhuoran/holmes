"""Encode without the per-plaintext device-to-host sync (``HOLMES_FAST_ENCODE=1``).

EasyFHE's ``kernels.cv_pre_encode_stage1`` ends with ``float(max_value.cpu().numpy())`` and ``encode_stage2`` checks
that scalar in ``_validate_scaled_range`` (max * scaling_factor < 2^61).  Holmes encodes ~1800 weight plaintexts per
layer online (``LinearWeight.plaintext`` -> ``ConstantBundle.plaintext`` -> stage1 + stage2), so the Python thread
blocks on the GPU ~1800 times per layer; ``plaintext_async`` keeps the encode on a side stream, but while the host
waits for that stream's max value it cannot enqueue the MAC kernels of the main stream (Sylph: ``Engine._encode_async``
drops exactly these two syncs).

The patch keeps the range check but defers it: ``cv_pre_encode_stage1`` returns the max as a device scalar wrapped in
:class:`DeferredMax`; ``_validate_scaled_range`` queues ``max * scaling_factor`` as a device scalar (no sync) and
:func:`check` reduces and reads them back once (the bench calls it after each layer and raises /
reports if any plaintext exceeded 2^61, exactly the condition EasyFHE checks eagerly).  Values that are consumed as a
Python float elsewhere (``float(deferred)``, comparisons) still sync lazily, so no behaviour changes outside the fast
path.  ``check()`` is also what the ``plaintext_async`` docstring's "one device-to-host read" turns into: zero reads
per plaintext, one per layer.  ``HOLMES_FAST_ENCODE=0`` leaves EasyFHE untouched.
"""
from __future__ import annotations

import math
import os

_installed = False
_pending: list = []    # device float64 scalars (max_encoded_value * scaling_factor), one per encode since last check


def enabled() -> bool:
    return os.environ.get("HOLMES_FAST_ENCODE", "0") == "1"


class DeferredMax:
    """Device scalar standing in for EasyFHE's ``max_encoded_value`` float; syncs only if read as a float."""

    __slots__ = ("tensor",)

    def __init__(self, tensor):
        self.tensor = tensor

    def __float__(self):
        return float(self.tensor.item())

    def __lt__(self, other):
        return float(self) < other

    def __le__(self, other):
        return float(self) <= other

    def __gt__(self, other):
        return float(self) > other

    def __ge__(self, other):
        return float(self) >= other

    def __mul__(self, other):
        return float(self) * other

    __rmul__ = __mul__

    def __repr__(self):
        return f"DeferredMax({self.tensor})"


def install() -> dict:
    global _installed
    if _installed:
        return {"fast_encode": True}
    import easyfhe as torch
    from easyfhe.fhe.ops import encoding, kernels

    def cv_pre_encode_stage1(input, slots, context):
        log_slots = int(math.log2(int(slots)))
        encoded, max_value = torch.fhe_pre_encode_stage1(
            input=input, slots=int(slots), M=context.M, rotGroup=context.encode_params_rotGroup,
            ksiPows=context.encode_params_ksiPows, bitrev=context.encode_bitrev_indices[log_slots])
        return encoded, DeferredMax(max_value.reshape(()).to(torch.float64))

    orig_validate = encoding._validate_scaled_range

    def _validate_scaled_range(prepared, scaling_factor):
        value = prepared.max_encoded_value
        if not isinstance(value, DeferredMax):
            return orig_validate(prepared, scaling_factor)
        # enqueued on the encode's own stream; reduced in check() after a device sync (encodes run on
        # side streams, so a running max across streams would race)
        _pending.append(value.tensor * float(scaling_factor))

    kernels.cv_pre_encode_stage1 = cv_pre_encode_stage1
    encoding._validate_scaled_range = _validate_scaled_range
    _installed = True
    return {"fast_encode": True}


def check(*, raise_on_overflow: bool = True) -> dict:
    """One device read: the worst ``max * scale`` since the last check (EasyFHE's limit is 2^61)."""

    global _pending
    import easyfhe as torch
    from easyfhe.fhe.ops.encoding import MAX_ENCODED_BITS

    if not _pending:
        return {"encodes": 0, "worst_bits": None, "ok": True}
    torch.cuda.synchronize()
    pending, _pending = _pending, []
    count = len(pending)
    # easyfhe's Tensor has no reductions; one D2H copy of the stacked scalars, max on the host
    worst = float(max(torch.stack(pending).cpu().tolist()))
    bits = math.log2(worst) if worst > 0 else None
    ok = bits is None or bits < MAX_ENCODED_BITS
    if not ok and raise_on_overflow:
        raise ValueError(f"fast encode: a plaintext exceeded the encoder range (max*scale = 2^{bits:.2f} over "
                         f"{count} encodes; limit 2^{MAX_ENCODED_BITS})")
    return {"encodes": count, "worst_bits": None if bits is None else round(bits, 2), "ok": ok}
