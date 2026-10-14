"""Kernel-level operation counter / GPU timer shared by the Holmes and Slot-Coeff (Sylph) benchmarks.

Two layers of counting, both attributed to the active ``region`` path:

* **kernel level** (``install``): every public callable of ``easyfhe.fhe.ops.kernels`` (``F.cv_*``) and of the
  extra modules (Sylph's ``keyswitch`` / ``rhombus`` / ... kernels) is wrapped; each call records ``calls``,
  ``cts`` (batch of the first ``[batch, limbs, N]`` tensor argument) and ``units`` (``batch * limbs``).  With
  ``OPCOUNT_TIME=1`` every call is bracketed by CUDA events (self time = inclusive minus nested wrapped calls).
  Kernel *calls* are not comparable across schemes (Sylph batches 16 ciphertexts into one launch, Holmes's BSGS
  rotations run batch 1); ``cts`` and ``units`` are.
* **API level** (``install_api_holmes`` / ``install_api_sylph``): the scheme-facing operations (rotate, CT x CT,
  PT x CT, bootstrap, S2C, HalfBoot, PCMM) are counted in *ciphertext-semantic* units under labels ``api.<op>``:
  one ``api.rot`` unit is one ciphertext rotated by one offset, one ``api.ctxct`` unit is one ciphertext
  product (+ relinearisation), one ``api.ptxct`` unit is one ciphertext x plaintext product, one
  ``api.bootstrap`` / ``api.halfboot`` / ``api.s2c`` unit is one ciphertext refreshed / converted.  A hoisted
  BSGS MAC (``hoisted_mac_sum``) counts ``baby - 1 + giant - 1`` rotations and ``baby * giant`` PT x CT; a
  batched Sylph ``rotate_many`` counts ``batch * len(steps)``.  While an API-level *leaf* is executing, nested
  API-level calls are not counted (``homo_rotate`` inside ``hoisted_mac_sum``, everything inside a bootstrap),
  so the API counts are the scheme's own operation counts, not EasyFHE's internal decomposition.  Non-leaf
  markers (``api.refresh``, ``api.poly``) count themselves and let their children count too.

Refresh points (bootstrap / S2C / HalfBoot) are additionally traced with input and output limbs and batch
(``take_trace()``), so both schemes' refresh schedules can be listed side by side.

``take()`` synchronises, returns ``{region: {label: {"calls", "cts", "units", "ms", "incl_ms"}}}`` and resets.

Enable with env ``OPCOUNT=1`` (counts) and ``OPCOUNT_TIME=1`` (counts + CUDA-event timing).
"""
from __future__ import annotations

import contextlib
import functools
import math
import os
import types

import easyfhe as torch

_installed = False
_time = False
_region: list[str] = ["-"]
_counts: dict = {}      # region -> label -> [calls, units, cts]
_events: list = []      # (region, label, ev0, ev1, parent)
_wrapped: list = []     # (owner, name, original)
_stack: list = []       # indices into _events of the wrapped calls currently executing (for self-time)
_api_depth = [0]        # > 0 while an API-level leaf is executing (nested API calls are not counted)
_trace: list = []       # refresh points: dict(region, op, cts, in_limbs, out_limbs)
TRACE_LABELS = ("bootstrap", "halfboot", "s2c")


def enabled() -> bool:
    return os.environ.get("OPCOUNT", "0") == "1" or os.environ.get("OPCOUNT_TIME", "0") == "1"


def timing() -> bool:
    return _time


# ----------------------------------------------------------------------------- helpers
def _batch_of_tensor(t):
    if t.dim() == 3:
        return int(t.shape[0]), int(t.shape[0]) * int(t.shape[1])
    if t.dim() == 2:
        return 1, int(t.shape[0])
    return 1, 1


def _volume(args, kwargs):
    """(cts, units) of the first tensor-like positional argument."""
    for a in args:
        if isinstance(a, torch.Tensor):
            return _batch_of_tensor(a)
        if isinstance(a, (list, tuple)) and a and isinstance(a[0], torch.Tensor):
            b, u = _batch_of_tensor(a[0])
            return len(a) * b, len(a) * u
        cv = getattr(a, "cv", None)      # Cipher-like objects
        if cv is not None and isinstance(cv[0], torch.Tensor):
            return _batch_of_tensor(cv[0])
    return 1, 1


def cts_of(x) -> int:
    """Ciphertext count of a Cipher / batched Cipher / list of them."""
    if x is None:
        return 0
    if isinstance(x, (list, tuple)):
        return sum(cts_of(v) for v in x)
    b = getattr(x, "batch_size", None)
    if b is not None:
        return int(b)
    cv = getattr(x, "cv", None)
    if cv is not None and isinstance(cv[0], torch.Tensor):
        return _batch_of_tensor(cv[0])[0]
    return 1


def limbs_of(x):
    """cur_limbs of a Cipher (first item of a list / tuple)."""
    while isinstance(x, (list, tuple)):
        if not x:
            return None
        x = x[0]
    st = getattr(x, "state", None)
    return None if st is None else int(st.cur_limbs)


def _add(label, calls=1, units=0, cts=0, reg=None):
    if reg is None:
        reg = "/".join(_region[1:]) or "-"
    slot = _counts.setdefault(reg, {}).setdefault(label, [0, 0, 0])
    slot[0] += calls
    slot[1] += units
    slot[2] += cts


def _wrap(label, f):
    @functools.wraps(f)
    def g(*args, **kwargs):
        cts, units = _volume(args, kwargs)
        _add(label, 1, units, cts)
        if _time:
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            parent = _stack[-1] if _stack else -1
            idx = len(_events)
            _events.append(("/".join(_region[1:]) or "-", label, e0, e1, parent))
            _stack.append(idx)
            e0.record()
            try:
                return f(*args, **kwargs)
            finally:
                e1.record()
                _stack.pop()
        return f(*args, **kwargs)
    g.__opcount_original__ = f
    return g


def wrap_module(mod, names=None, prefix=""):
    """Wrap the given (or all public) functions of ``mod`` in place (kernel level)."""
    if names is None:
        names = [n for n, v in vars(mod).items()
                 if not n.startswith("_") and isinstance(v, types.FunctionType) and v.__module__ == mod.__name__]
    for n in names:
        f = getattr(mod, n, None)
        if f is None or hasattr(f, "__opcount_original__"):
            continue
        _wrapped.append((mod, n, f))
        setattr(mod, n, _wrap(prefix + n, f))


def wrap_method(cls, name, label):
    f = getattr(cls, name)
    if hasattr(f, "__opcount_original__"):
        return
    _wrapped.append((cls, name, f))
    setattr(cls, name, _wrap(label, f))


def wrap_region(owner, name, label):
    """Make calls to ``owner.name`` open a region named ``label`` (kernels inside are attributed to it)."""
    f = getattr(owner, name)
    if getattr(f, "__opcount_region__", None) == label:
        return

    @functools.wraps(f)
    def g(*args, **kwargs):
        with region(label):
            return f(*args, **kwargs)
    g.__opcount_region__ = label
    g.__opcount_inner__ = f
    _wrapped.append((owner, name, f))
    setattr(owner, name, g)


@contextlib.contextmanager
def region(label):
    if _region[-1] == label:            # halfboots -> halfboot etc.: do not repeat the label in the path
        yield
        return
    _region.append(label)
    try:
        yield
    finally:
        _region.pop()


# ----------------------------------------------------------------------------- API level
def _api_region(label):
    """Region path for an API record; a trailing region equal to the op itself (``.../bootstrap`` opened by the
    region wrapper around the same call) is dropped so refresh ops are charged to the operator calling them."""
    reg = list(_region[1:])
    if reg and reg[-1] == label:
        reg = reg[:-1]
    return "/".join(reg) or "-"


def wrap_api(owner, name, label, units=None, *, leaf=True, trace=False, method=False, aliases=()):
    """Count calls of ``owner.name`` as ``api.<label>`` with ``units(*args, **kwargs)`` ciphertext-semantic
    units (default: batch of the first cipher argument).  ``units`` may return a dict ``{label: n}`` to
    charge several labels (BSGS MAC -> rot + ptxct).  ``leaf`` suppresses nested API counts.  ``trace``
    records (in_limbs, out_limbs, cts) for refresh-type ops.  ``method``: ``args[0]`` is ``self``.
    ``aliases`` are other ``(owner, name)`` pairs bound to the same function object that must be redirected
    to the wrapper too (``easyfhe.fhe.homo_rotate`` is ``rotation.homo_rotate``).  ``label=None`` makes a
    silent leaf (counts nothing, only hides the nested calls of a compound EasyFHE helper)."""
    f = getattr(owner, name)
    if hasattr(f, "__opcount_api__"):
        return f

    @functools.wraps(f)
    def g(*args, **kwargs):
        counted = _api_depth[0] == 0 and label is not None
        if counted:
            reg = _api_region(label)
            try:
                u = units(*args, **kwargs) if units is not None else cts_of(args[1 if method else 0])
            except Exception:   # noqa: BLE001  - never let bookkeeping break the computation
                u = 1
            if isinstance(u, dict):
                for k, v in u.items():
                    if v:
                        _add("api." + k, 1 if k == label else 0, int(v), int(v), reg=reg)
            else:
                _add("api." + label, 1, int(u), int(u), reg=reg)
        if leaf:
            _api_depth[0] += 1
        try:
            out = f(*args, **kwargs)
        finally:
            if leaf:
                _api_depth[0] -= 1
        if trace and counted:
            src = args[1 if method else 0]
            _trace.append({"region": reg, "op": label, "cts": cts_of(src),
                           "in_limbs": limbs_of(src), "out_limbs": limbs_of(out)})
        return out
    g.__opcount_api__ = label
    g.__opcount_original__ = f
    _wrapped.append((owner, name, f))
    setattr(owner, name, g)
    for o2, n2 in aliases:
        if getattr(o2, n2, None) is f:
            _wrapped.append((o2, n2, f))
            setattr(o2, n2, g)
    return g


def _nonzero(offsets):
    if isinstance(offsets, (int, float)):
        return 1 if int(offsets) else 0
    return len(set(int(o) for o in offsets if int(o)))


def _n_plain(p):
    b = getattr(p, "batch_size", None)
    if b is not None:
        return int(b)
    try:
        return len(p)
    except TypeError:
        return 1


def install_api_holmes():
    """API-level counters for Holmes: ``easyfhe.fhe`` namespace (what ``crypto_context.fhe.*`` resolves to), the
    ``easyfhe.fhe.ops`` modules behind it (same function objects, so ``fhe.homo_rotate`` and
    ``rotation.homo_rotate`` share one wrapper) and ``easyfhe.bs.openfhe.bootstrap``.  Every other public
    function of the ops modules becomes a silent leaf, so e.g. ``align_to`` does not leak its inner rescale."""
    from easyfhe import fhe
    from easyfhe.fhe.ops import alignment, arithmetic, rotation
    mods = (fhe, arithmetic, rotation, alignment)

    def hook(name, label, units=None, leaf=True):
        for m in mods:
            if getattr(m, name, None) is not None:
                wrap_api(m, name, label, units, leaf=leaf, aliases=[(x, name) for x in mods if x is not m])
                return

    def mac_units(cipher, baby_offsets, plaintexts, giant_offset, giant_count, context, *, baby_rotations=None, **kw):
        baby = len(tuple(baby_offsets))
        rot = (max(0, baby - 1) if baby_rotations is None else 0) + (max(0, int(giant_count) - 1) if int(giant_offset) else 0)
        return {"rot": rot * cts_of(cipher), "ptxct": _n_plain(plaintexts) * cts_of(cipher), "bsgs_mac": cts_of(cipher)}

    hook("homo_rotate", "rot", lambda c, off, ctx, *a, **k: cts_of(c) * _nonzero(off))
    hook("fast_rotate", "rot", lambda c, offs, ctx, *a, **k: cts_of(c) * _nonzero(offs))
    hook("prepare_hoisted_baby_rotations", "rot", lambda c, offs, ctx, *a, **k: cts_of(c) * _nonzero(offs))
    hook("hoisted_mac_sum", "bsgs_mac", mac_units)
    hook("hoisted_mac_sum_rescale", "bsgs_mac", mac_units)
    hook("grouped_pairwise_mac", "ptxct", lambda cs, pts, groups, ctx, *a, **k: _n_plain(pts))
    hook("homo_mul_pt", "ptxct")
    hook("homo_mul_pt_rescale", "ptxct")
    hook("homo_mul_relin", "ctxct")
    hook("homo_mul_relin_rescale_postop", "ctxct")
    hook("homo_mul_no_relin", "ctxct_norelin")
    hook("homo_relinearize", "relin")
    hook("homo_mul_scalar", "scalar")
    hook("homo_mul_i", "scalar")
    hook("homo_add", "add")
    hook("homo_sub", "add")
    hook("homo_add_pt", "add_pt")
    hook("rescale", "rescale")
    for m in (arithmetic, rotation, alignment):          # compound helpers: hide their internal calls
        for n, v in list(vars(m).items()):
            if n.startswith("_") or not isinstance(v, types.FunctionType) or v.__module__ != m.__name__:
                continue
            if hasattr(v, "__opcount_api__"):
                continue
            wrap_api(m, n, None, aliases=[(fhe, n)])
    try:
        import easyfhe.bs.openfhe as bs
        import easyfhe.bs.openfhe.api as api
        impl = getattr(api.bootstrap, "__opcount_inner__", api.bootstrap)   # strip the region wrapper of install()
        if not hasattr(impl, "__opcount_api__"):
            api.bootstrap = impl
            wrap_api(api, "bootstrap", "bootstrap", lambda c, *a, **k: cts_of(c), trace=True)
            wrap_region(api, "bootstrap", "bootstrap")      # region outermost, then the api counter, then impl
            bs.bootstrap = api.bootstrap
    except Exception:
        pass


def install_api_sylph(engine_cls, kit_cls, rhombus_cls):
    """API-level counters for Slot-Coeff: ``Engine`` slot ops, ``CoefficientKit`` S2C / HalfBoot and
    ``Rhombus`` PCMM.  Units are ciphertexts (batch items); ``pcmm`` units are output coefficient cts and
    its ``ptxct`` units are ct x pt products (automorphic copies x weight plaintexts)."""
    E, K, R = engine_cls, kit_cls, rhombus_cls

    def rot_step(self, a, step, *r, **k):
        return cts_of(a) if int(step) % self.slots else 0

    def n_steps(self, a, steps):
        return cts_of(a) * len(set(int(s) % self.slots for s in steps) - {0})

    M = dict(method=True)
    wrap_api(E, "mul", "ctxct", lambda self, a, b, *r, **k: max(cts_of(a), cts_of(b)), **M)
    wrap_api(E, "cmul", "ptxct", lambda self, a, *r, **k: cts_of(a), **M)
    wrap_api(E, "cmul_sum", "ptxct", lambda self, terms, *r, **k: len(terms) * max(cts_of(t[0]) for t in terms), **M)
    wrap_api(E, "add_vector", "add_pt", lambda self, a, *r, **k: cts_of(a), **M)
    wrap_api(E, "rotate", "rot", rot_step, **M)
    wrap_api(E, "rotate_add", "rot", lambda self, addend, a, step, *r, **k: rot_step(self, a, step), **M)
    wrap_api(E, "rotations", "rot", lambda self, a, steps, *r, **k: n_steps(self, a, steps), **M)
    wrap_api(E, "rotate_many", "rot", lambda self, a, steps, *r, **k: n_steps(self, a, steps), **M)
    wrap_api(E, "smul", "scalar", lambda self, a, *r, **k: cts_of(a), **M)
    wrap_api(E, "add", "add", lambda self, a, b, *r, **k: max(cts_of(a), cts_of(b)), **M)
    wrap_api(E, "sub", "add", lambda self, a, b, *r, **k: max(cts_of(a), cts_of(b)), **M)
    wrap_api(E, "refresh", "refresh", lambda self, ct, *r, **k: cts_of(ct), leaf=False, **M)
    wrap_api(E, "poly", "poly", lambda self, x, coeffs, *r, **k: cts_of(x), leaf=False, **M)
    wrap_api(K, "to_coefficients", "s2c", lambda self, z, *r, **k: 1, trace=True, **M)
    wrap_api(K, "to_coefficients_many", "s2c", lambda self, zs, *r, **k: len(zs), trace=True, **M)
    wrap_api(K, "halfboot", "halfboot", lambda self, c, *r, **k: 1, trace=True, **M)
    wrap_api(K, "halfboots", "halfboot", lambda self, cs, *r, **k: len(cs), trace=True, **M)

    def pcmm_units(self, inputs, weights, split, *r, **k):
        plan = self.plan(split)
        rows = weights.dev.shape[0] if hasattr(weights, "dev") else int(weights.shape[0])
        outs = rows // self.k
        # every automorphic input copy is multiplied by pc weight plaintexts (one per packed output group),
        # so ct x pt products = out blocks x (input blocks x A) x pc = out blocks x input blocks x K
        return {"pcmm": outs, "ptxct": outs * sum(cts_of(x) for x in inputs) * int(plan["pc"]),
                "pcmm_rot": outs * int(math.log2(plan["pc"])) if plan["pc"] > 1 else 0}

    wrap_api(R, "project", "pcmm", pcmm_units, **M)
    wrap_api(R, "inputs", "pcmm_rot", lambda self, blocks, split, *r, **k: len(blocks) * (self.plan(split)["a"] - 1), **M)


def take_trace():
    """Return and clear the refresh-point trace (list of dict(region, op, cts, in_limbs, out_limbs))."""
    global _trace
    out, _trace = _trace, []
    return out


# ----------------------------------------------------------------------------- install / take
def install(*, time: bool | None = None, extra_modules=()):
    """Wrap ``easyfhe.fhe.ops.kernels`` (all ``cv_*`` / ``cipher_*``) plus ``extra_modules``.

    Call this *after* any framework monkey-patching of the kernels module (e.g. Sylph's
    ``keyswitch.install``) so the wrappers see the functions that will actually run."""
    global _installed, _time
    if time is None:
        time = os.environ.get("OPCOUNT_TIME", "0") == "1"
    _time = bool(time)
    if _installed:
        return
    _installed = True
    from easyfhe.fhe.ops import kernels as F
    names = [n for n in dir(F) if (n.startswith("cv_") or n.startswith("cipher_")) and callable(getattr(F, n))]
    wrap_module(F, names, prefix="F.")
    for mod in extra_modules:
        wrap_module(mod, prefix=mod.__name__.rsplit(".", 1)[-1] + ".")
    try:   # Holmes / EasyFHE bootstrap as a region
        import easyfhe.bs.openfhe as bs
        import easyfhe.bs.openfhe.api as api
        wrap_region(api, "bootstrap", "bootstrap")
        bs.bootstrap = api.bootstrap
    except Exception:
        pass


def take():
    """Synchronise, return ``{region: {label: {calls, cts, units, ms, incl_ms}}}`` since the last call and reset.

    ``ms`` is exclusive (self) GPU time: inclusive time minus the inclusive time of wrapped calls nested
    inside, so sums over kernels equal the total wrapped GPU time without double counting.  ``api.*`` labels
    carry counts only (``ms`` 0)."""
    global _counts, _events
    torch.cuda.synchronize()
    out = {}
    for reg, ks in _counts.items():
        out[reg] = {k: {"calls": v[0], "units": v[1], "cts": v[2], "ms": 0.0, "incl_ms": 0.0} for k, v in ks.items()}
    incl = [float(e0.elapsed_time(e1)) for (_, _, e0, e1, _) in _events]
    child = [0.0] * len(_events)
    for i, (_, _, _, _, parent) in enumerate(_events):
        if parent >= 0:
            child[parent] += incl[i]
    for i, (reg, k, _, _, _) in enumerate(_events):
        out[reg][k]["ms"] += incl[i] - child[i]
        out[reg][k]["incl_ms"] += incl[i]
    for reg in out:
        for k in out[reg]:
            out[reg][k]["ms"] = round(out[reg][k]["ms"], 3)
            out[reg][k]["incl_ms"] = round(out[reg][k]["incl_ms"], 3)
    _counts, _events = {}, []
    _stack.clear()
    return out


FIELDS = ("calls", "units", "cts", "ms", "incl_ms")


def merge(a, b):
    """Add snapshot ``b`` into ``a`` (both as returned by ``take``)."""
    for reg, ks in b.items():
        for k, v in ks.items():
            s = a.setdefault(reg, {}).setdefault(k, {f: 0.0 for f in FIELDS})
            for f in FIELDS:
                s[f] = s.get(f, 0.0) + v.get(f, 0.0)
    return a


def summarize(snap):
    """Collapse regions: ``{label: {calls, units, cts, ms}}`` plus ``{"_regions": {region: ms}}``."""
    tot, regs = {}, {}
    for reg, ks in snap.items():
        for k, v in ks.items():
            s = tot.setdefault(k, {f: 0.0 for f in FIELDS})
            for f in FIELDS:
                s[f] = s.get(f, 0.0) + v.get(f, 0.0)
            regs[reg] = regs.get(reg, 0.0) + v["ms"]
    tot["_regions"] = regs
    return tot
