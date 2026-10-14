from __future__ import annotations

"""Shifted-lane (SL) decode packing and its numpy BSGS oracle.

Decode carries one token in a feature-major ciphertext whose eight token
lanes are ``slot(c, l) = c * lanes + l``.  The lane-replicated (LR) form used
until now fills lane 0 only; a diagonal linear then multiplies 7/8 of every
plaintext slot by zero and needs ``dimension`` diagonals per square block.

The SL form puts the eight *circular shifts* of the same vector into the
eight lanes::

    SL:  value(slot(c, l)) = x[(c + shift * l) mod D],   shift = D / lanes

A feature rotation by ``k`` (``k * lanes`` slots) keeps the lane structure,
so after ``shift`` baby/giant diagonals lane ``l`` of channel ``c`` holds the
partial product of output ``c`` over the input block ``[shift*l, shift*(l+1))``
and three lane-fold rotations (1, 2, 4 slots) finish the sum in lane 0.  The
diagonal count per 4096x4096 block drops from 4096 to 512 (GAZELLE's hybrid
matrix-vector product).  Lanes 1..7 of the folded output hold cross-channel
garbage and must be masked before the vector re-enters SL form.

Rotation semantics follow EasyFHE's ``homo_rotate`` (source minus
destination): ``rotate(v, r)[s] = v[(s + r) mod N]`` i.e. ``np.roll(v, -r)``.
"""

from dataclasses import dataclass

import numpy as np

from .linear import _positive_power_of_two


def rotate(vector: np.ndarray, offset: int) -> np.ndarray:
    """EasyFHE rotation oracle: ``out[s] = vector[(s + offset) mod N]``."""

    return np.roll(np.asarray(vector), -int(offset))


@dataclass(frozen=True)
class ShiftedLaneLayout:
    """One-token shifted-lane carrier (``lanes = slots / hidden_dim``)."""

    hidden_dim: int = 4096
    slots: int = 32768

    def __post_init__(self) -> None:
        _positive_power_of_two(self.hidden_dim, name="hidden_dim")
        _positive_power_of_two(self.slots, name="slots")
        if int(self.slots) % int(self.hidden_dim):
            raise ValueError(
                "shifted-lane layout requires slots divisible by hidden_dim, "
                f"got slots={self.slots}, hidden_dim={self.hidden_dim}."
            )
        if int(self.hidden_dim) % self.lanes:
            raise ValueError(
                f"hidden_dim={self.hidden_dim} must be divisible by lanes={self.lanes}."
            )

    @property
    def lanes(self) -> int:
        return int(self.slots) // int(self.hidden_dim)

    @property
    def shift(self) -> int:
        """Channel shift between neighbouring lanes (= diagonals per block)."""

        return int(self.hidden_dim) // self.lanes

    def slot(self, channel: int, lane: int) -> int:
        channel = int(channel)
        lane = int(lane)
        if not 0 <= channel < int(self.hidden_dim):
            raise IndexError(f"channel={channel} is outside [0, {self.hidden_dim}).")
        if not 0 <= lane < self.lanes:
            raise IndexError(f"lane={lane} is outside [0, {self.lanes}).")
        return channel * self.lanes + lane

    # ------------------------------------------------------------- vectors
    def channel_of_slot(self) -> np.ndarray:
        return np.arange(int(self.slots), dtype=np.int64) // self.lanes

    def lane_of_slot(self) -> np.ndarray:
        return np.arange(int(self.slots), dtype=np.int64) % self.lanes

    def lane_mask(self, lane: int = 0) -> np.ndarray:
        """Plaintext mask selecting one lane (1.0 there, 0.0 elsewhere)."""

        return (self.lane_of_slot() == int(lane)).astype(np.float64)

    def pack(self, x: np.ndarray, *, dtype=np.float64) -> np.ndarray:
        """SL carrier of ``x`` (shape ``(hidden_dim,)``)."""

        x = np.asarray(x, dtype=dtype).reshape(-1)
        if x.shape != (int(self.hidden_dim),):
            raise ValueError(
                f"shifted-lane pack expects shape {(int(self.hidden_dim),)}, got {x.shape}."
            )
        source = (self.channel_of_slot() + self.shift * self.lane_of_slot()) % int(
            self.hidden_dim
        )
        return x[source]

    def pack_lane0(self, x: np.ndarray, *, dtype=np.float64) -> np.ndarray:
        """LR carrier of ``x``: lane 0 holds ``x``, other lanes are zero."""

        x = np.asarray(x, dtype=dtype).reshape(-1)
        if x.shape != (int(self.hidden_dim),):
            raise ValueError(
                f"lane-0 pack expects shape {(int(self.hidden_dim),)}, got {x.shape}."
            )
        out = np.zeros((int(self.slots),), dtype=x.dtype)
        out[:: self.lanes] = x
        return out

    def unpack(self, carrier: np.ndarray, *, lane: int = 0) -> np.ndarray:
        """Read ``x`` back from one lane (undoing that lane's shift)."""

        carrier = np.asarray(carrier)
        if carrier.shape != (int(self.slots),):
            raise ValueError(
                f"shifted-lane carrier must have shape {(int(self.slots),)}, got {carrier.shape}."
            )
        lane = int(lane)
        shifted = carrier[lane :: self.lanes]
        return np.roll(shifted, self.shift * lane)

    def is_shifted_lane(self, carrier: np.ndarray, *, atol: float = 1e-9) -> bool:
        """Whether every lane is the correctly shifted copy of lane 0."""

        reference = self.unpack(carrier, lane=0)
        return all(
            np.allclose(self.unpack(carrier, lane=lane), reference, atol=atol)
            for lane in range(1, self.lanes)
        )

    # --------------------------------------------------------- conversions
    def lane0_to_shifted_offsets(self) -> tuple[int, ...]:
        """Rotation offsets that build the SL form from a lane-0-only carrier.

        ``slot(c, l)`` must receive ``x[c + shift*l]`` which sits at
        ``slot(c + shift*l, 0)``: source minus destination is
        ``shift*l*lanes - l = (hidden_dim - 1) * l``.
        """

        return tuple((int(self.hidden_dim) - 1) * lane for lane in range(1, self.lanes))

    def lane0_to_shifted(self, carrier: np.ndarray, *, masked: bool = True) -> np.ndarray:
        """LR -> SL oracle: optional lane-0 mask, then ``lanes - 1`` rotations.

        ``masked=False`` assumes lanes 1..7 are already zero (e.g. the RMSNorm
        gamma plaintext carried the mask) and skips the level-consuming mask.
        """

        carrier = np.asarray(carrier)
        base = carrier * self.lane_mask(0) if masked else carrier
        out = base.copy()
        for offset in self.lane0_to_shifted_offsets():
            out = out + rotate(base, offset)
        return out

    def fold_offsets(self) -> tuple[int, ...]:
        """Lane-fold rotations (1, 2, 4, ...) summing the lanes into lane 0."""

        return tuple(1 << i for i in range(int(np.log2(self.lanes))))

    def fold_lanes(self, carrier: np.ndarray) -> np.ndarray:
        """Sum the lanes of each channel into lane 0 (lanes 1.. hold garbage)."""

        out = np.asarray(carrier)
        for offset in self.fold_offsets():
            out = out + rotate(out, offset)
        return out


# ---------------------------------------------------------------- BSGS oracle
def diagonal_plaintext(
    weight: np.ndarray,
    *,
    layout: ShiftedLaneLayout,
    baby: int,
    giant: int,
    baby_steps: int,
    lane_shift: int,
) -> np.ndarray:
    """One BSGS plaintext row ``row(g, b)``.

    ``row(g,b)[slot(c,l)] = W[(c - g*B) mod D, (c + b + lane_shift*l) mod D]``.
    ``lane_shift=0`` is the existing lane-replicated packing (``D`` diagonals);
    ``lane_shift=layout.shift`` is the SL packing (``shift`` diagonals).
    """

    dimension = int(layout.hidden_dim)
    channel = layout.channel_of_slot()
    lane = layout.lane_of_slot()
    output_feature = (channel - int(giant) * int(baby_steps)) % dimension
    input_feature = (channel + int(baby) + int(lane_shift) * lane) % dimension
    return np.asarray(weight)[output_feature, input_feature]


def bsgs_matvec(
    weight: np.ndarray,
    carrier: np.ndarray,
    *,
    layout: ShiftedLaneLayout,
    baby_steps: int,
    lane_shift: int,
    giant_indices=None,
) -> np.ndarray:
    """BSGS oracle over ``diagonals = D`` (LR) or ``shift`` (SL) diagonals.

    ``giant_indices`` restricts the giant steps (a rank's chunk); summing the
    results over a partition of the giants reproduces the full product, which
    is exactly what the decode all-reduce does.  The result is *unfolded*: for
    the SL packing call :meth:`ShiftedLaneLayout.fold_lanes` on the (summed)
    output, for LR the lanes already carry the full product.
    """

    dimension = int(layout.hidden_dim)
    baby_steps = int(baby_steps)
    diagonals = dimension if int(lane_shift) == 0 else int(lane_shift)
    if diagonals % baby_steps:
        raise ValueError(f"baby_steps={baby_steps} must divide diagonals={diagonals}.")
    giant_steps = diagonals // baby_steps
    giants = range(giant_steps) if giant_indices is None else tuple(int(g) for g in giant_indices)
    carrier = np.asarray(carrier)
    babies = [rotate(carrier, baby * layout.lanes) for baby in range(baby_steps)]
    out = np.zeros_like(carrier, dtype=np.result_type(carrier, np.asarray(weight)))
    for giant in giants:
        acc = np.zeros_like(out)
        for baby in range(baby_steps):
            acc += babies[baby] * diagonal_plaintext(
                weight,
                layout=layout,
                baby=baby,
                giant=giant,
                baby_steps=baby_steps,
                lane_shift=lane_shift,
            )
        out += rotate(acc, giant * baby_steps * layout.lanes)
    return out


def shifted_lane_linear(
    weight: np.ndarray,
    x: np.ndarray,
    *,
    layout: ShiftedLaneLayout,
    baby_steps: int,
) -> np.ndarray:
    """End-to-end SL oracle: ``x`` (lane-0 carrier) -> ``W @ x`` in lane 0."""

    carrier = layout.lane0_to_shifted(layout.pack_lane0(x))
    product = bsgs_matvec(
        weight,
        carrier,
        layout=layout,
        baby_steps=baby_steps,
        lane_shift=layout.shift,
    )
    return layout.fold_lanes(product)


__all__ = [
    "ShiftedLaneLayout",
    "bsgs_matvec",
    "diagonal_plaintext",
    "rotate",
    "shifted_lane_linear",
]
