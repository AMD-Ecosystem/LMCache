# SPDX-License-Identifier: Apache-2.0
"""
Tests for the L1 ``mlock`` option: ``L1MemoryManagerConfig.mlock``,
``--l1-mlock`` / ``--no-l1-mlock`` and the ``mlock`` key of ``--l1-manager``.

Per the docstrings and help text, the default is on for ROCm and off for other
platforms, and both the command line and the JSON config can override it.
"""

# Standard
import json

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.config import L1MemoryManagerConfig, parse_args
from lmcache.v1.platform import current_device_spec

LEGACY_ARGS = ["--l1-size-gb", "1", "--eviction-policy", "LRU"]


def _json_args(**fields: object) -> list[str]:
    spec = {"type": "DRAM", "tag": "_default", "size_gb": 1, **fields}
    return ["--eviction-policy", "LRU", "--l1-manager", json.dumps(spec)]


def _legacy_mlock(extra: list[str]) -> bool:
    return parse_args(LEGACY_ARGS + extra).l1_manager_config.memory_config.mlock


def _json_mlock(**fields: object) -> bool:
    return parse_args(_json_args(**fields)).l1_manager_config.memory_config.mlock


@pytest.fixture(params=["rocm", "cuda"])
def backend(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> str:
    """Make the current platform report ``request.param`` as its backend."""
    monkeypatch.setattr(
        type(current_device_spec),
        "backend_name",
        property(lambda self: request.param),
    )
    return request.param


def test_mlock_defaults_on_for_rocm_only(backend: str) -> None:
    expected = backend == "rocm"
    config = L1MemoryManagerConfig(size_in_bytes=1 << 30, use_lazy=True)
    assert config.mlock is expected
    assert _legacy_mlock([]) is expected
    assert _json_mlock() is expected


@pytest.mark.parametrize(
    "flag, expected", [("--l1-mlock", True), ("--no-l1-mlock", False)]
)
def test_command_line_overrides_default(
    backend: str, flag: str, expected: bool
) -> None:
    assert _legacy_mlock([flag]) is expected


@pytest.mark.parametrize("value", [True, False])
def test_json_key_overrides_default(backend: str, value: bool) -> None:
    assert _json_mlock(mlock=value) is value


def test_mlock_is_accepted_without_lazy_allocation(backend: str) -> None:
    assert _legacy_mlock(["--no-l1-use-lazy", "--l1-mlock"]) is True
    assert _json_mlock(use_lazy=False, mlock=True) is True


def test_json_mlock_must_be_boolean() -> None:
    with pytest.raises(ValueError, match="boolean"):
        parse_args(_json_args(mlock="true"))
