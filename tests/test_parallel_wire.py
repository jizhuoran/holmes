"""CPU protocol tests for the ciphertext wire format (fake tensors, no EasyFHE)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from llama3fhe.parallel.wire import (
    CipherPacket,
    cipher_metadata,
    pack_cipher,
    rebuild_cipher,
    unpack_cipher,
    validate_cipher,
)


class _Tensor:
    """Fake device tensor: ``cpu().numpy()`` like EasyFHE, ``to()`` for moves."""

    def __init__(self, device="cuda:0", shape=(1, 4, 8), dtype="torch.uint64", payload=0):
        self.device = device
        self.shape = tuple(shape)
        self.dtype = dtype
        self.payload = payload

    def to(self, device):
        if device == self.device:
            return self
        return _Tensor(device, self.shape, self.dtype, self.payload)

    def clone(self):
        return _Tensor(self.device, self.shape, self.dtype, self.payload)

    def cpu(self):
        return self.to("cpu")

    def numpy(self):
        if self.device != "cpu":
            raise TypeError("can't convert cuda tensor to numpy")
        return np.full(self.shape, self.payload, dtype=np.uint64)


class _Cipher:
    def __init__(self, device="cuda:0", limbs=3, shape=(1, 4, 8), payload=(1, 2)):
        self.cv = [_Tensor(device, shape, payload=payload[0]), _Tensor(device, shape, payload=payload[1])]
        self.state = SimpleNamespace(cur_limbs=limbs, scale_degree=1, scaling_factor=2.0**40)
        self.slots = 32768
        self.is_ext = False
        self.batch_size = 1

    def cipher_like(self, components):
        clone = _Cipher.__new__(_Cipher)
        clone.cv = list(components)
        clone.state = self.state
        clone.slots = self.slots
        clone.is_ext = self.is_ext
        clone.batch_size = self.batch_size
        return clone


def test_metadata_captures_state_and_size() -> None:
    metadata = cipher_metadata(_Cipher())
    assert metadata.component_count == 2
    assert metadata.cur_limbs == 3
    assert metadata.scaling_factor == 2.0**40
    assert metadata.device == "cuda:0"
    assert metadata.nbytes == 2 * 4 * 8 * 8
    assert metadata.without_devices() == cipher_metadata(_Cipher(device="cpu")).without_devices()


def test_pack_stages_numpy_on_host_and_unpack_rebuilds_on_device(monkeypatch) -> None:
    source = _Cipher(payload=(7, 9))
    packet = pack_cipher(source, tag="attn_in/lane/3")
    assert isinstance(packet, CipherPacket)
    assert packet.tag == "attn_in/lane/3"
    # host side is plain NumPy (the only thing multiprocessing can pickle)
    assert all(isinstance(component, np.ndarray) for component in packet.components)
    assert [int(c[0, 0, 0]) for c in packet.components] == [7, 9]
    assert packet.nbytes == 2 * 4 * 8 * 8
    # the source is untouched
    assert all(component.device == "cuda:0" for component in source.cv)

    # receiver side: easyfhe.from_numpy(...).to(device) is stubbed with the fake tensor
    import sys
    import types

    fake_easyfhe = types.ModuleType("easyfhe")

    def from_numpy(array):
        return _Tensor("cpu", array.shape, "torch." + str(array.dtype), int(array.flat[0]))

    fake_easyfhe.from_numpy = from_numpy
    monkeypatch.setitem(sys.modules, "easyfhe", fake_easyfhe)

    rebuilt = unpack_cipher(packet, device="cuda", template=_Cipher())
    assert all(component.device == "cuda" for component in rebuilt.cv)
    assert [c.payload for c in rebuilt.cv] == [7, 9]
    assert cipher_metadata(rebuilt).without_devices() == packet.metadata.without_devices()


def test_unpack_without_template_uses_packet_state(monkeypatch) -> None:
    import sys
    import types

    fake_easyfhe = types.ModuleType("easyfhe")
    fake_easyfhe.from_numpy = lambda array: _Tensor("cpu", array.shape, "torch." + str(array.dtype), int(array.flat[0]))
    monkeypatch.setitem(sys.modules, "easyfhe", fake_easyfhe)

    packet = pack_cipher(_Cipher(limbs=2))
    rebuilt = unpack_cipher(packet, device="cpu")
    assert rebuilt.state.cur_limbs == 2
    assert rebuilt.slots == 32768
    packet.state = None
    with pytest.raises(ValueError):
        unpack_cipher(packet, device="cpu")


def test_packet_round_trips_through_pickle() -> None:
    import pickle

    packet = pack_cipher(_Cipher(payload=(3, 4)), tag="pairfm/partial/2")
    clone = pickle.loads(pickle.dumps(packet, protocol=pickle.HIGHEST_PROTOCOL))
    assert clone.tag == packet.tag
    assert clone.metadata == packet.metadata
    assert [int(c[0, 0, 0]) for c in clone.components] == [3, 4]
    assert clone.state.cur_limbs == 3


def test_rebuild_never_aliases_and_validate_rejects_bad_ciphers() -> None:
    source = _Cipher()
    rebuilt = rebuild_cipher(source, [c.clone() for c in source.cv])
    assert rebuilt is not source
    assert rebuilt.cv[0] is not source.cv[0]

    mixed = _Cipher()
    mixed.cv[1] = _Tensor("cuda:1")
    with pytest.raises(ValueError, match="multiple devices"):
        validate_cipher(mixed)
    too_many_limbs = _Cipher(limbs=9)
    with pytest.raises(ValueError, match="exceeds"):
        validate_cipher(too_many_limbs)
    with pytest.raises(ValueError, match="does not match"):
        validate_cipher(_Cipher(), expected_device="cuda:3")
    validate_cipher(_Cipher(), expected_device="cuda")

    bad_packet = pack_cipher(_Cipher())
    bad_packet.components = bad_packet.components[:1]
    with pytest.raises(ValueError):
        bad_packet.validate()
