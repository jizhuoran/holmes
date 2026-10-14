"""``HOLMES_SPARSE_FUSED=1``: fused MAC + batched giant step for the attention sparse BSGS scheduler.



This module replaces that inner loop with, per encode batch:

* one gathered MAC launch (``holmes_mac_gather`` in bench/cmp/kernels/holmes_mac.cu): every group of every
  input in one kernel, babies read in place through a per-row index (no selection copy), output
  ``res[2][K][inputs][limbs][N]``;
* one batched ModUp of all (group, input) partial sums + one ``sylph_giant_sum_ext`` launch that rotates
  each group by its own giant offset and adds everything into an *extended* accumulator per input
  (Sylph's double-hoisted giant step; no per-group ModDown).

Groups with giant offset 0 (and the schedule's padding rows) are added without rotation.  The extended
accumulators are ModDown'ed once per input at the end, with the running totals folded into that ModDown.
Same message as the per-group path (one ModDown rounding per input instead of one per group), same
logical PT x CT / rotation counts; residues differ by ModDown rounding.

Requires ``HOLMES_SPARSE_NONEXT_BABY=1`` (non-ext baby bases), Sylph's key-switch kernels
(``HOLMES_SYLPH_KERNELS=1``) and the holmes_mac library (bench/cmp on sys.path, as in bench/t2_model.py);
otherwise :func:`create` returns ``None`` and the scheduler keeps its loop.
"""

from __future__ import annotations

import ctypes
import os

import numpy as np

from llama3fhe.backend import release_if_supported

ENABLED = os.environ.get("HOLMES_SPARSE_FUSED", "0") not in ("", "0", "false", "no")
STATS = {"calls": 0, "batches": 0, "groups": 0, "zero_groups": 0, "fallbacks": 0}


def _modules():
    """(holmes_multi_mac, sylph keyswitch, sylph bsgs) or None when any piece is unavailable."""
    try:
        import holmes_multi_mac  # bench/cmp (sys.path of bench/t2_model.py)
        from sylph_easy.easyfhe_ext import bsgs, keyswitch
    except ImportError:
        return None
    if not keyswitch.enabled():
        return None
    return holmes_multi_mac, keyswitch, bsgs


class SparseFusedAccumulator:
    """Fused MAC + ext-domain giant accumulation for one sparse scheduler call (one baby basis per input)."""

    def __init__(self, mods, baby_batches, baby_index, crypto_context):
        import easyfhe as torch

        self._torch = torch
        self.mac, self.ks, self.bsgs = mods
        self.lib = self.mac.load()
        self.crypto_context = crypto_context
        self.ctx = ctx = crypto_context.context
        self.baby_index = dict(baby_index)
        self.inputs = len(baby_batches)
        self.template = baby_batches[0]
        self.limbs = int(self.template.state.cur_limbs)
        self.n, self.k, self.L, self.alpha = int(ctx.N), int(ctx.K), int(ctx.L), int(ctx.alpha)
        self.beta = (self.limbs + self.alpha - 1) // self.alpha
        layouts = [self.mac._layout(b, self.limbs, self.n) for b in baby_batches]
        if any(lay is None for lay in layouts):
            raise ValueError("baby bases must be non-ext contiguous [nb][limbs][N] batches")
        self.bstride = layouts[0][2]
        if any(lay[2] != self.bstride for lay in layouts):
            raise ValueError("baby bases must share their item stride")
        self.nb = int(self.template.batch_size)
        ptr_t = ctypes.c_void_p * 32
        self._c0s = ptr_t(*[lay[0].data_ptr() for lay in layouts])
        self._c1s = ptr_t(*[lay[1].data_ptr() for lay in layouts])
        self.ti = 4 if self.inputs >= 4 else (2 if self.inputs >= 2 else 1)
        self.device = layouts[0][0].device
        self.active = self.limbs + self.k
        self.acc = None  # [2][inputs][limbs + K][N] extended accumulator (None until a rotated group lands)
        self._spare = None
        self.state = None

    @classmethod
    def create(cls, baby_batches, baby_index, crypto_context, schedule):
        if not ENABLED or not baby_batches:
            return None
        if any(bool(b.is_ext) for b in baby_batches) or len(baby_batches) > 32:
            return cls._fallback("ext baby bases (needs HOLMES_SPARSE_NONEXT_BABY=1) or > 32 inputs")
        mods = _modules()
        if mods is None:
            return cls._fallback("holmes_multi_mac / sylph keyswitch unavailable")
        limbs = {int(b.state.cur_limbs) for b in baby_batches}
        alpha = int(crypto_context.context.alpha)
        # giant_sum_ext holds <= 3 digits per thread (Holmes's patched copy: 5, bench/cmp/holmes_sylph_kernel_src.py)
        if len(limbs) != 1 or (next(iter(limbs)) + alpha - 1) // alpha > 5:
            return cls._fallback(f"limbs {sorted(limbs)} / alpha {alpha}: too many digits")
        try:
            acc = cls(mods, baby_batches, baby_index, crypto_context)
        except ValueError as exc:
            return cls._fallback(f"layout: {exc}")
        acc.prepare(schedule)
        STATS["calls"] += 1
        return acc

    @staticmethod
    def _fallback(reason: str):
        STATS["fallbacks"] += 1
        STATS["reason"] = reason
        if os.environ.get("HOLMES_SPARSE_FUSED_WARN", "1") != "0" and STATS["fallbacks"] == 1:
            import warnings

            warnings.warn(f"HOLMES_SPARSE_FUSED: falling back to the scheduler loop ({reason})", RuntimeWarning, stacklevel=3)
        return None

    # -- per encode batch -------------------------------------------------------------------------------------
    def prepare(self, schedule):
        """Precompute every encode batch's (baby index, plaintext row, segment) tables: one H2D copy per call
        instead of one synchronous pageable copy per batch (which would serialise the host with the device)."""
        torch = self._torch
        plans = []
        chunks = []
        cursor = 0
        for batch in schedule.batches:
            groups, zero = [], []
            start = 0
            for group in batch.groups:
                stop = start + int(group.row_count)
                idx = [self.baby_index[int(o)] for o in group.baby_offsets]
                (groups if int(group.giant_offset) else zero).append((int(group.giant_offset), idx, start, stop))
                start = stop
            if int(batch.padding_rows):
                stop = start + int(batch.padding_rows)
                zero.append((0, [self.baby_index[0]] * int(batch.padding_rows), start, stop))
            nk = len(groups)
            groups += zero
            K = len(groups)
            idx_host = np.concatenate([np.asarray(g[1], dtype=np.int32) for g in groups])
            prow_host = np.concatenate([np.arange(g[2], g[3], dtype=np.int32) for g in groups])
            seg_host = np.zeros(K + 1, dtype=np.int32)
            seg_host[1:] = np.cumsum([len(g[1]) for g in groups])
            rows = int(idx_host.size)
            if prow_host.size != rows or rows != int(batch.encoded_rows):
                raise ValueError("route rows and baby indices disagree")
            chunk = np.concatenate([idx_host, prow_host, seg_host])
            chunks.append(chunk)
            plans.append((K, nk, tuple(g[0] for g in groups[:nk]), rows, cursor))
            cursor += int(chunk.size)
        self._meta = torch.from_numpy(np.concatenate(chunks)).to(self.device)
        self._plans = plans

    def apply_batch(self, batch_index, plaintexts, totals, add_owned):
        """MAC + giant step of one encode batch; zero-offset groups and padding rows go into ``totals``."""
        torch = self._torch
        from easyfhe.fhe.ciphertext import CipherState

        K, nk, offsets, rows, cursor = self._plans[int(batch_index)]
        if K == 0:
            return
        meta = self._meta
        idx_dev = meta[cursor: cursor + rows]
        prow_dev = meta[cursor + rows: cursor + 2 * rows]
        seg_dev = meta[cursor + 2 * rows: cursor + 2 * rows + K + 1]
        pv = plaintexts.cv[0]
        pv = pv if pv.dim() == 3 else pv[None]
        if pv.stride(1) != self.n or pv.stride(2) != 1 or pv.size(1) < self.limbs or pv.size(0) < rows:
            raise ValueError("unexpected plaintext layout")
        pstride = int(pv.stride(0))
        res = torch.empty((2, K, self.inputs, self.limbs, self.n), dtype=torch.uint64, device=self.device)
        stream = torch.cuda.current_stream().cuda_stream
        code = self.lib.holmes_mac_gather(
            res.data_ptr(), self._c0s, self._c1s, self.bstride, idx_dev.data_ptr(), prow_dev.data_ptr(),
            seg_dev.data_ptr(), self.inputs, K, pv.data_ptr(), pstride, self.ti, self.limbs, self.ctx.primes.data_ptr(),
            self.ctx.barret_ratio.data_ptr(), self.ctx.barret_k.data_ptr(), stream)
        if code:
            raise RuntimeError(f"holmes_mac_gather failed with code {code}")
        if self.state is None:
            self.state = CipherState(self.limbs, self.template.state.scale_degree + plaintexts.state.scale_degree,
                                     self.template.state.scaling_factor * plaintexts.state.scaling_factor)
            self.slots = plaintexts.slots
        STATS["batches"] += 1
        STATS["groups"] += K
        STATS["zero_groups"] += K - nk
        if nk:
            self._giant_sum(res, nk, offsets)
        for k in range(nk, K):
            for i in range(self.inputs):
                term = self.template.cipher_like([res[0, k, i].clone(), res[1, k, i].clone()], state=self.state,
                                                 slots=self.slots, is_ext=False, batch_size=1)
                totals[i] = add_owned(totals[i], term, crypto_context=self.crypto_context)
        del res

    def _giant_sum(self, res, nk, offsets):
        torch = self._torch
        from sylph_easy.easyfhe_ext import cudalib

        ks, ctx = self.ks, self.ctx
        c0 = res[0, :nk].reshape(nk * self.inputs, self.limbs, self.n)
        c1 = res[1, :nk].reshape(nk * self.inputs, self.limbs, self.n)
        digits = ks.modup(c1, self.limbs, ctx)
        table = ks._fast_table(ctx, offsets, self.limbs)
        if self.acc is None:
            self.acc = torch.zeros((2, self.inputs, self.active, self.n), dtype=torch.uint64, device=self.device)
        if self._spare is None:
            self._spare = torch.empty_like(self.acc)
        nxt = self._spare
        lib = self.bsgs._lib()
        cudalib.check(lib.sylph_giant_sum_ext(
            nxt[0].data_ptr(), nxt[1].data_ptr(), self.acc[0].data_ptr(), self.acc[1].data_ptr(), digits.data_ptr(),
            self.beta * self.active * self.n, c0.data_ptr(), self.limbs * self.n, table.data_ptr(), nk, self.inputs,
            self.limbs, self.k, self.L, self.beta, ctx.primes.data_ptr(), ctx.barret_ratio.data_ptr(),
            ctx.barret_k.data_ptr(), ctx.PModq.data_ptr(), torch.cuda.current_stream().cuda_stream), "giant_sum_ext")
        self._spare, self.acc = self.acc, nxt
        del digits

    # -- end of the scheduler call ----------------------------------------------------------------------------
    def finish(self, totals, add_owned):
        """ModDown the extended accumulators once per input, folding the running totals in."""
        torch = self._torch
        if self.acc is None:
            return totals
        add0 = add1 = None
        fold = all(t is not None for t in totals)
        if fold:
            add0 = torch.stack([t.cv[0] if t.cv[0].dim() == 2 else t.cv[0][0] for t in totals], 0)
            add1 = torch.stack([t.cv[1] if t.cv[1].dim() == 2 else t.cv[1][0] for t in totals], 0)
            if add0.size(1) != self.limbs:
                add0, add1 = add0[:, : self.limbs].contiguous(), add1[:, : self.limbs].contiguous()
        cur0, cur1 = self.ks.moddown_pair(self.acc[0], self.acc[1], self.limbs, self.ctx, add0, add1)
        self.acc = self._spare = None
        del add0, add1
        out = []
        for i in range(self.inputs):
            cipher = self.template.cipher_like([cur0[i], cur1[i]], state=self.state, slots=self.slots, is_ext=False,
                                               batch_size=1)
            if fold:
                release_if_supported(totals[i])
                out.append(cipher)
            else:
                out.append(add_owned(totals[i], cipher, crypto_context=self.crypto_context))
        return out
