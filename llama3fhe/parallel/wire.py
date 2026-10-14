"""Wire format for moving a CKKS ciphertext between T2 worker processes.

An EasyFHE :class:`~easyfhe.fhe.ciphertext.Cipher` is a list of per-prime
residue tensors (``cv``) plus a frozen :class:`CipherState`
(``cur_limbs``, ``scale_degree``, ``scaling_factor``) and the encoding
contract (``slots``, ``is_ext``, ``batch_size``).  Nothing in it may be
reinterpreted on the way: a raw NCCL/torch collective on ``cv`` is not a
valid CKKS operation, and only the *owner* rank may add partials
(:func:`llama3fhe.operators.attention.distributed.distributed_output.pairfm_target_reduce_fhe`).

Hence the wire object is deliberately dumb:



``pack_cipher`` / ``unpack_cipher`` accept anything that quacks like a
Cipher (``cv``, ``state``, ``slots``, ``is_ext``, ``batch_size``,
``cipher_like``) so the protocol has CPU tests with small fake tensors.
"""

from __future__ import annotations

import copy
import importlib
from dataclasses import dataclass
from typing import Iterable

import numpy as np

_ELEMENT_BYTES = {
    "uint64": 8,
    "int64": 8,
    "float64": 8,
    "complex128": 16,
    "uint32": 4,
    "int32": 4,
    "float32": 4,
    "complex64": 8,
    "uint16": 2,
    "int16": 2,
    "float16": 2,
    "bfloat16": 2,
    "uint8": 1,
    "int8": 1,
    "bool": 1,
}


def _device_string(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _shape(value: object) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        return ()
    try:
        return tuple(int(extent) for extent in shape)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"cipher component has invalid shape {shape!r}") from exc


def _components(cipher: object) -> tuple[object, ...]:
    components = getattr(cipher, "cv", None)
    if components is None:
        raise TypeError("cipher must expose a non-empty cv component sequence")
    if hasattr(components, "dim"):
        components = (components,)
    try:
        components = tuple(components)
    except TypeError as exc:
        raise TypeError("cipher.cv must be iterable") from exc
    if not components:
        raise ValueError("cipher.cv must not be empty")
    return components


@dataclass(frozen=True)
class CipherMetadata:
    """Everything except the residues.

    ``component_shapes`` includes the full allocated limb dimension, not just
    the active ``cur_limbs`` prefix; EasyFHE kernels may rely on the capacity.
    """

    component_shapes: tuple[tuple[int, ...], ...]
    component_dtypes: tuple[str | None, ...]
    component_devices: tuple[str | None, ...]
    cur_limbs: int
    scale_degree: int
    scaling_factor: float | None
    slots: int
    batch_size: int
    is_ext: bool

    @property
    def component_count(self) -> int:
        return len(self.component_shapes)

    @property
    def device(self) -> str | None:
        devices = {device for device in self.component_devices if device is not None}
        if len(devices) == 1:
            return next(iter(devices))
        return None

    @property
    def nbytes(self) -> int:
        """Best-effort payload byte count (zero for unknown dtypes)."""

        total = 0
        for shape, dtype in zip(self.component_shapes, self.component_dtypes, strict=True):
            if dtype is None:
                continue
            element_bytes = _ELEMENT_BYTES.get(dtype.removeprefix("torch."))
            if element_bytes is None:
                continue
            elements = 1
            for extent in shape:
                elements *= int(extent)
            total += elements * element_bytes
        return int(total)

    def without_devices(self) -> tuple[object, ...]:
        """Comparison key that ignores where the components live."""

        return (
            self.component_shapes,
            self.component_dtypes,
            self.cur_limbs,
            self.scale_degree,
            self.scaling_factor,
            self.slots,
            self.batch_size,
            self.is_ext,
        )


def cipher_metadata(cipher: object) -> CipherMetadata:
    """Extract and validate the wire metadata from ``cipher``."""

    components = _components(cipher)
    state = getattr(cipher, "state", None)
    if state is None:
        raise TypeError("cipher must expose state metadata")
    try:
        cur_limbs = int(state.cur_limbs)
        scale_degree = int(state.scale_degree)
    except (AttributeError, TypeError, ValueError) as exc:
        raise TypeError("cipher.state must expose integer limb/scale fields") from exc
    if cur_limbs <= 0 or scale_degree <= 0:
        raise ValueError(
            f"cipher state must be positive, got cur_limbs={cur_limbs}, "
            f"scale_degree={scale_degree}"
        )
    scaling_factor = getattr(state, "scaling_factor", None)
    if scaling_factor is not None:
        scaling_factor = float(scaling_factor)
    slots = int(getattr(cipher, "slots"))
    batch_size = int(getattr(cipher, "batch_size", 1))
    if slots <= 0 or batch_size <= 0:
        raise ValueError(f"slots/batch_size must be positive, got {slots}/{batch_size}")
    return CipherMetadata(
        component_shapes=tuple(_shape(component) for component in components),
        component_dtypes=tuple(
            None if getattr(component, "dtype", None) is None else str(component.dtype)
            for component in components
        ),
        component_devices=tuple(
            _device_string(getattr(component, "device", None)) for component in components
        ),
        cur_limbs=cur_limbs,
        scale_degree=scale_degree,
        scaling_factor=scaling_factor,
        slots=slots,
        batch_size=batch_size,
        is_ext=bool(getattr(cipher, "is_ext", False)),
    )


def validate_cipher(cipher: object, *, expected_device: str | None = None) -> CipherMetadata:
    """Check basic Cipher invariants and return the metadata."""

    metadata = cipher_metadata(cipher)
    devices = {device for device in metadata.component_devices if device is not None}
    if len(devices) > 1:
        raise ValueError(f"cipher components span multiple devices: {sorted(devices)}")
    if expected_device is not None and metadata.device is not None:
        wanted = str(expected_device)
        actual = metadata.device
        same = actual == wanted or (
            wanted == "cuda" and actual.startswith("cuda")
        ) or (actual == "cuda" and wanted.startswith("cuda"))
        if not same:
            raise ValueError(f"cipher device {actual!r} does not match {wanted!r}")
    for shape in metadata.component_shapes:
        if shape and metadata.batch_size != int(shape[0]):
            raise ValueError(
                "cipher batch_size does not match component shape: "
                f"{metadata.batch_size} vs {shape[0]}"
            )
        if len(shape) >= 2 and metadata.cur_limbs > int(shape[-2]):
            raise ValueError(
                f"cipher cur_limbs={metadata.cur_limbs} exceeds component capacity {shape[-2]}"
            )
    return metadata


def rebuild_cipher(template: object, components: Iterable[object]) -> object:
    """Wrap ``components`` with ``template``'s state/encoding metadata.

    Prefers ``Cipher.cipher_like``; falls back to constructing
    ``easyfhe.fhe.Cipher`` directly, and finally to a shallow copy for
    duck-typed fakes in CPU tests.  ``template`` is never mutated.
    """

    components = list(components)
    cipher_like = getattr(template, "cipher_like", None)
    if callable(cipher_like):
        return cipher_like(components)
    try:
        fhe = importlib.import_module("easyfhe.fhe")
        return fhe.Cipher(
            components,
            template.state,
            int(template.slots),
            bool(getattr(template, "is_ext", False)),
            int(getattr(template, "batch_size", 1)),
        )
    except Exception:
        clone = copy.copy(template)
        clone.cv = components
        return clone


def _to_host_array(component: object) -> object:
    """Stage one ``cv`` component on the host as a NumPy array (always a copy).

    Real EasyFHE tensors go through ``.cpu().numpy()``; NumPy arrays are
    copied; duck-typed fakes without either are cloned when possible.
    """

    if isinstance(component, np.ndarray):
        return np.array(component, copy=True)
    numpy = getattr(component, "numpy", None)
    if callable(numpy):
        cpu = getattr(component, "cpu", None)
        host = cpu() if callable(cpu) else component
        detach = getattr(host, "detach", None)
        if callable(detach):
            host = detach()
        return np.array(host.numpy(), copy=True)
    clone = getattr(component, "clone", None)
    return clone() if callable(clone) else component


def _to_device_tensor(component: object, device: str) -> object:
    """Rebuild a device tensor from a host component.

    NumPy arrays become EasyFHE tensors on ``device``; tensor-like objects
    are moved with ``.to(device)``; anything else is returned unchanged (CPU
    tests with fakes).
    """

    if isinstance(component, np.ndarray):
        easyfhe = importlib.import_module("easyfhe")
        tensor = easyfhe.from_numpy(component)
        return tensor.to(str(device)) if str(device) != "cpu" else tensor
    to = getattr(component, "to", None)
    if callable(to):
        moved = to(str(device))
        if moved is component:
            clone = getattr(component, "clone", None)
            if callable(clone):
                moved = clone()
        return moved
    return component


@dataclass
class CipherPacket:
    """A ciphertext staged on the host, ready for an inter-process queue."""

    metadata: CipherMetadata
    components: tuple[object, ...]
    #: Free-form routing tag (``"attn_in/lane/3"`` etc.).  Not interpreted here.
    tag: str = ""
    #: Cipher state object (picklable frozen dataclass in EasyFHE); kept so a
    #: receiver without a template can rebuild the exact wrapper.
    state: object | None = None

    @property
    def nbytes(self) -> int:
        return self.metadata.nbytes

    def validate(self) -> None:
        if len(self.components) != self.metadata.component_count:
            raise ValueError("packet component count does not match metadata")
        for component, shape in zip(self.components, self.metadata.component_shapes, strict=True):
            if _shape(component) != shape:
                raise ValueError("packet component shape does not match metadata")


def pack_cipher(cipher: object, *, tag: str = "", host: bool = True) -> CipherPacket:
    """Record the metadata and stage every ``cv`` component for the transport.

    ``host=True`` (default) copies the components to the host as NumPy arrays,
    which every backend can move.  ``host=False`` keeps them as device tensors
    **by reference** for a transport whose ``device_payloads`` is true (the
    pinned arena copies them straight off the device); the transport converts
    them to the host anyway if it cannot.  ``cipher`` is not consumed either
    way; the caller decides when to release it -- after the collective that
    carried the packet returned, since with ``host=False`` the copy happens
    inside that collective.
    """

    metadata = validate_cipher(cipher)
    if host:
        components = tuple(_to_host_array(component) for component in _components(cipher))
    else:
        components = tuple(_components(cipher))
    packet = CipherPacket(
        metadata=metadata,
        components=components,
        tag=str(tag),
        state=getattr(cipher, "state", None),
    )
    packet.validate()
    return packet


class _PacketTemplate:
    """Minimal duck-typed Cipher so ``rebuild_cipher`` can work without one."""

    def __init__(self, packet: CipherPacket) -> None:
        self.state = packet.state
        self.slots = int(packet.metadata.slots)
        self.is_ext = bool(packet.metadata.is_ext)
        self.batch_size = int(packet.metadata.batch_size)
        self.cv: list[object] = []


def unpack_cipher(
    packet: CipherPacket, *, device: str = "cuda", template: object | None = None
) -> object:
    """Rebuild a Cipher on ``device`` from the packet's components.

    Components are host NumPy arrays (``"host"`` backend) or device tensors
    the transport already restored on ``device`` (``"pinned"`` backend, or the
    in-process emulation handing them over by reference); the latter are
    cloned so the rebuilt cipher never aliases the sender's storage.  Inside a
    T2 worker ``device`` is always the exact string ``"cuda"`` (the launcher
    pins one physical card through ``CUDA_VISIBLE_DEVICES``).  When
    ``template`` is ``None`` the packet's own state is used, which requires
    ``easyfhe.fhe.Cipher`` unless the state came from a fake cipher.
    """

    packet.validate()
    components = tuple(_to_device_tensor(component, str(device)) for component in packet.components)
    base = template if template is not None else _PacketTemplate(packet)
    if template is None and packet.state is None:
        raise ValueError("packet has no state; pass a template cipher")
    rebuilt = rebuild_cipher(base, components)
    after = validate_cipher(rebuilt, expected_device=str(device))
    if after.without_devices() != packet.metadata.without_devices():
        raise RuntimeError(
            "cipher metadata changed during transport: "
            f"sent={packet.metadata.without_devices()!r} got={after.without_devices()!r}"
        )
    return rebuilt


__all__ = [
    "CipherMetadata",
    "CipherPacket",
    "cipher_metadata",
    "pack_cipher",
    "rebuild_cipher",
    "unpack_cipher",
    "validate_cipher",
]
