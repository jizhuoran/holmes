"""E3: run Holmes on Slot-Coeff's hand-written EasyFHE kernels (kernel attribution of the layer-time gap).

Sylph's ``easyfhe_ext`` patches module-level functions of ``easyfhe.fhe.ops`` (``kernels.cv_keyswitch`` /
``cv_modup`` / ``cv_rescale_one_level`` / ``cv_hmul_relin_rescale``, ``arithmetic.homo_mul_relin`` /
``homo_mul_relin_rescale_postop``, ``rotation.fast_rotate`` / ``moddown_from_ext``, ``alignment._rescale_one_level``).
Holmes calls the same operations through the ``easyfhe.fhe`` namespace, whose names were bound at import time, so
after installing we re-point every ``easyfhe.fhe.<name>`` that the patches replaced.  Nothing else of Holmes
changes: same context (depth 37, dnum 5, BS (5,5)/17), same schedule, same ciphertexts.

Env: ``HOLMES_SYLPH_KERNELS=1`` (read by bench/t2_model.py), ``SYLPH_ROOT`` (dir containing ``sylph_easy``,
default ``./third_party/sylph-easy``), ``HOLMES_SYLPH_HOSTCACHE=1`` to also install Sylph's host-side plan/key caches,
plus Sylph's own ``SYLPH_KS / SYLPH_HMUL / SYLPH_FASTROT / SYLPH_EVALMOD=0`` switches to ablate one kernel at a time.
Correctness check: ``final_hidden_error`` of the t2_model run must match the plain Holmes run.
"""
from __future__ import annotations

import os
import sys


def install(*, hostcache: bool | None = None) -> dict:
    root = os.environ.get("SYLPH_ROOT", "./third_party/sylph-easy")
    if root not in sys.path:
        sys.path.insert(0, root)
    from easyfhe import fhe
    from easyfhe.fhe.ops import alignment, arithmetic, kernels, rotation
    from sylph_easy.easyfhe_ext import evalmod, keyswitch

    kernel_dir = None
    if os.environ.get("HOLMES_BATCH_BS", "0") == "1" or os.environ.get("HOLMES_WIDE_BSGS", "0") == "1":
        # Sylph's BSGS kernels take <= 3 key-switch digits (dnum 3); Holmes's dnum-5 tower needs 5 at the raised
        # modulus.  Load the kernels from Holmes's patched copy of the sources (bench/cmp/holmes_sylph_kernel_src.py).
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import holmes_sylph_kernel_src
        kernel_dir = holmes_sylph_kernel_src.prepare(root)

    before = {m: dict(vars(m)) for m in (arithmetic, rotation, alignment, kernels)}
    if hostcache is None:
        hostcache = os.environ.get("HOLMES_SYLPH_HOSTCACHE", "0") == "1"
    if hostcache:
        from sylph_easy.easyfhe_ext import hostcache as hc
        hc.install()
    keyswitch.install()
    evalmod.install()

    changed = {}
    for m, old in before.items():
        for name, cur in vars(m).items():
            if callable(cur) and old.get(name) is not cur and name in old:
                changed[f"{m.__name__.rsplit('.', 1)[-1]}.{name}"] = True
                # the fhe namespace re-exports the ops functions by value; follow the patch
                if getattr(fhe, name, None) is old[name]:
                    setattr(fhe, name, cur)
                    changed[f"fhe.{name}"] = True
    return {"keyswitch": keyswitch.enabled(), "evalmod": evalmod.enabled(), "hostcache": bool(hostcache),
            "kernel_dir": kernel_dir, "patched": sorted(changed), "modules": [keyswitch, evalmod]}


if __name__ == "__main__":
    print({k: v for k, v in install().items() if k != "modules"})
