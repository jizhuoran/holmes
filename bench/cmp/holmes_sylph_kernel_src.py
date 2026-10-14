"""Holmes's copy of the Sylph CUDA sources, with the BSGS kernels widened to 5 key-switch digits."""
from __future__ import annotations

import os
import re
import shutil
import sys

MAX_BETA = 5


def _replace_exact(src: str, old: str, new: str, count: int) -> str:
    found = src.count(old)
    if found != count:
        raise RuntimeError(f"Sylph keyswitch.cu changed: expected {count} x {old!r}, found {found}")
    return src.replace(old, new)


def patch_keyswitch(src: str) -> str:
    # MB template parameter on both BSGS kernels
    src = _replace_exact(src, "template <int TB, int G>\n__global__ void __launch_bounds__(256, 2) bsgs_mac_ext(",
                         "template <int TB, int G, int MB>\n__global__ void __launch_bounds__(256, 2) bsgs_mac_ext(", 1)
    src = _replace_exact(src, "template <int TB>\n__global__ void __launch_bounds__(256) giant_sum_ext(",
                         "template <int TB, int MB>\n__global__ void __launch_bounds__(256) giant_sum_ext(", 1)
    src = _replace_exact(src, "    uint64_t kb[3], ka[3];", "    uint64_t kb[MB], ka[MB];", 2)
    src = _replace_exact(src, "for (int d = 0; d < 3; ++d) {", "for (int d = 0; d < MB; ++d) {", 4)
    src = _replace_exact(src, "  if (beta > 3) return -1;", f"  if (beta > {MAX_BETA}) return -1;", 2)
    # giant_sum_ext launch: one statement -> dispatch on beta
    m = re.search(r"  giant_sum_ext<TB><<<(.*?)\);\n", src, re.S)
    if m is None or src.count("giant_sum_ext<TB><<<") != 1:
        raise RuntimeError("Sylph keyswitch.cu changed: giant_sum_ext launch not found")
    stmt = m.group(0)
    src = src.replace(stmt, "  if (beta <= 3) {\n" + stmt.replace("giant_sum_ext<TB><<<", "giant_sum_ext<TB, 3><<<")
                      + "  } else {\n" + stmt.replace("giant_sum_ext<TB><<<", f"giant_sum_ext<TB, {MAX_BETA}><<<")
                      + "  }\n")
    # bsgs_mac_ext launch macro: BSGS(TB, G) -> dispatch on beta
    src = _replace_exact(src, "#define BSGS(TB, G)", "#define BSGS1(TB, G, MB)", 1)
    src = _replace_exact(src, "bsgs_mac_ext<TB, G><<<", "bsgs_mac_ext<TB, G, MB><<<", 1)
    src = _replace_exact(src, "#undef BSGS\n",
                         "#undef BSGS\n#undef BSGS1\n", 1)
    # insert the dispatching BSGS macro right after the BSGS1 macro definition (ends at the first line without '\')
    i = src.index("#define BSGS1(TB, G, MB)")
    j = i
    while True:
        k = src.index("\n", j)
        if src[k - 1] != "\\":
            break
        j = k + 1
    src = src[:k + 1] + f"#define BSGS(TB, G) do {{ if (beta <= 3) BSGS1(TB, G, 3); else BSGS1(TB, G, {MAX_BETA}); }} while (0)\n" + src[k + 1:]
    return src


def prepare(sylph_root: str | None = None, cache_dir: str | None = None) -> str:
    """Write the patched kernel sources and point ``cudalib`` at them; returns the directory."""

    sylph_root = sylph_root or os.environ.get("SYLPH_ROOT", "./third_party/sylph-easy")
    if sylph_root not in sys.path:
        sys.path.insert(0, sylph_root)
    from sylph_easy.easyfhe_ext import cudalib, keyswitch

    if getattr(keyswitch, "_lib", None) is not None:
        raise RuntimeError("Sylph kernels already loaded; prepare() must run before holmes_sylph_kernels.install()")
    src_dir = os.path.join(sylph_root, "sylph_easy", "easyfhe_ext", "kernels")
    cache_dir = cache_dir or os.environ.get("HOLMES_SYLPH_KERNEL_DIR",
                                            os.path.join(os.path.expanduser("~"), ".cache", "holmes_sylph_kernels"))
    os.makedirs(cache_dir, exist_ok=True)
    for name in sorted(os.listdir(src_dir)):
        if not name.endswith(".cu"):
            continue
        with open(os.path.join(src_dir, name), "r") as f:
            text = f.read()
        if name == "keyswitch.cu":
            text = patch_keyswitch(text)
        dst = os.path.join(cache_dir, name)
        if not os.path.exists(dst) or open(dst).read() != text:
            with open(dst, "w") as f:
                f.write(text)
    cudalib._DIR = cache_dir
    return cache_dir


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("SYLPH_ROOT", "./third_party/sylph-easy")
    with open(os.path.join(root, "sylph_easy", "easyfhe_ext", "kernels", "keyswitch.cu")) as f:
        out = patch_keyswitch(f.read())
    sys.stdout.write(out)
