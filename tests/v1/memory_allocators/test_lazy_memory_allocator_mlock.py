# SPDX-License-Identifier: Apache-2.0
"""
Tests for the ``mlock`` option of LazyMemoryAllocator.

Per the docstrings:
- With ``mlock=True`` the initial chunk is locked by the constructor and every
  expansion chunk before it is committed, so ``VmLck`` grows to the full size,
  and the whole buffer is excluded from transparent huge pages.
- ``close()`` unlocks the memory again.
- A failed ``mlock`` logs one warning plus a summary; the memory is still
  allocated and usable.
- ``warn_if_compaction_moves_mlocked_pages`` warns unless the sysctl is ``0``
  and stays silent when the file cannot be read.
"""

# Standard
from pathlib import Path
import logging
import os
import resource
import subprocess
import sys
import textwrap
import time

# Third Party
import pytest

# First Party
from lmcache import torch_dev
from lmcache.v1.memory_allocators.lazy_memory_allocator import (
    LazyMemoryAllocator,
    warn_if_compaction_moves_mlocked_pages,
)
from lmcache.v1.platform import current_device_spec

pytestmark = pytest.mark.no_shared_allocator

MB = 1 << 20
INIT_SIZE = 128 * MB
FINAL_SIZE = 512 * MB

requires_pinned_memory = pytest.mark.skipif(
    not (torch_dev.is_available() and current_device_spec.is_pin_supported),
    reason="Requires a GPU backend with pinned-memory support",
)
requires_unlimited_memlock = pytest.mark.skipif(
    resource.getrlimit(resource.RLIMIT_MEMLOCK)[0] != resource.RLIM_INFINITY,
    reason="Requires an unlimited RLIMIT_MEMLOCK (docker run --ulimit memlock=-1)",
)

_LOGGER_NAME = "lmcache.v1.memory_allocators.lazy_memory_allocator"

# Runs in a subprocess so the lowered RLIMIT_MEMLOCK and the dropped
# CAP_IPC_LOCK (which root, e.g. in a container, has and which bypasses the
# limit) do not leak into the test process.
_LIMITED_MEMLOCK_CHILD = textwrap.dedent(
    f"""
    import ctypes, resource

    import torch

    from lmcache import torch_device_type
    from lmcache.v1.memory_allocators.lazy_memory_allocator import (
        LazyMemoryAllocator,
    )

    libc = ctypes.CDLL(None, use_errno=True)
    header = (ctypes.c_uint32 * 2)(0x20080522, 0)  # _LINUX_CAPABILITY_VERSION_3
    caps = (ctypes.c_uint32 * 6)()  # 2 x (effective, permitted, inheritable)
    assert libc.capget(header, caps) == 0
    caps[0] &= ~(1 << 14)  # CAP_IPC_LOCK, effective set
    assert libc.capset(header, caps) == 0
    resource.setrlimit(resource.RLIMIT_MEMLOCK, (64 << 10, 64 << 10))

    allocator = LazyMemoryAllocator({INIT_SIZE}, {FINAL_SIZE}, mlock=True)
    obj = allocator.allocate(torch.Size([{MB}]), torch.uint8)
    obj.tensor.fill_(7)
    assert torch.equal(obj.tensor.to(torch_device_type).cpu(), obj.tensor)
    allocator.free(obj)
    allocator.close()
    print("CHILD_OK")
    """
)


def _vm_lck_bytes() -> int:
    """Return this process's locked memory (``VmLck``) in bytes."""
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmLck:"):
            return int(line.split()[1]) * 1024
    raise AssertionError("VmLck not found in /proc/self/status")


def _vm_flags(ptr: int) -> list[str]:
    """Return the ``VmFlags`` of the mapping in /proc/self/smaps containing ptr."""
    inside = False
    for line in Path("/proc/self/smaps").read_text().splitlines():
        first = line.split()[0]
        if not first.endswith(":"):
            start, end = (int(x, 16) for x in first.split("-"))
            inside = start <= ptr < end
        elif inside and first == "VmFlags:":
            return line.split()[1:]
    raise AssertionError(f"no mapping contains {ptr:#x}")


def _wait_until_expanded(allocator: LazyMemoryAllocator, size: int) -> None:
    """Wait for the background expansion to commit ``size`` bytes."""
    deadline = time.monotonic() + 60
    while allocator.get_address_manager().get_heap_size() < size:
        assert time.monotonic() < deadline, "expansion did not finish in 60 s"
        time.sleep(0.05)


class _ListHandler(logging.Handler):
    """Collect records of the allocator logger (it does not propagate)."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def allocator_warnings():
    """Yield the WARNING records logged by the allocator module."""
    handler = _ListHandler()
    module_logger = logging.getLogger(_LOGGER_NAME)
    original_level = module_logger.level
    module_logger.setLevel(logging.WARNING)
    module_logger.addHandler(handler)
    try:
        yield handler.records
    finally:
        module_logger.removeHandler(handler)
        module_logger.setLevel(original_level)


@requires_pinned_memory
@requires_unlimited_memlock
def test_mlock_locks_initial_and_expanded_memory_until_close():
    before = _vm_lck_bytes()
    allocator = LazyMemoryAllocator(INIT_SIZE, FINAL_SIZE, mlock=True)
    try:
        assert _vm_lck_bytes() - before >= INIT_SIZE
        _wait_until_expanded(allocator, FINAL_SIZE)
        assert _vm_lck_bytes() - before >= FINAL_SIZE
    finally:
        allocator.close()
    assert _vm_lck_bytes() - before < INIT_SIZE


@requires_pinned_memory
def test_without_mlock_memory_is_not_locked():
    before = _vm_lck_bytes()
    allocator = LazyMemoryAllocator(INIT_SIZE, FINAL_SIZE, mlock=False)
    try:
        _wait_until_expanded(allocator, FINAL_SIZE)
        assert _vm_lck_bytes() - before < INIT_SIZE
    finally:
        allocator.close()


@requires_pinned_memory
@pytest.mark.parametrize("mlock", [True, False])
def test_mlock_excludes_buffer_from_transparent_huge_pages(mlock: bool):
    allocator = LazyMemoryAllocator(INIT_SIZE, FINAL_SIZE, mlock=mlock)
    try:
        buffer = allocator.get_underlying_buffer()
        first, last = buffer.data_ptr(), buffer.data_ptr() + buffer.numel() - 1
        # "nh" = VM_NOHUGEPAGE, set by madvise(MADV_NOHUGEPAGE)
        assert ("nh" in _vm_flags(first)) is mlock
        assert ("nh" in _vm_flags(last)) is mlock
    finally:
        allocator.close()


@requires_pinned_memory
def test_mlock_failure_warns_once_and_memory_stays_usable():
    result = subprocess.run(
        [sys.executable, "-c", _LIMITED_MEMLOCK_CHILD],
        capture_output=True,
        text=True,
        timeout=300,
        env={**os.environ, "LMCACHE_LOG_LEVEL": "INFO"},
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "CHILD_OK" in result.stdout
    # One warning for the first failed chunk, one summary counting the
    # initial chunk and every expansion chunk.
    failed = 1 + (FINAL_SIZE - INIT_SIZE) // LazyMemoryAllocator.PIN_CHUNK_SIZE
    assert output.count("mlock failed for chunk") == 1, output
    assert output.count(f"mlock failed for {failed} chunks") == 1, output


@pytest.mark.parametrize("value", ["1\n", "2\n"])
def test_sysctl_check_warns_when_compaction_may_move_locked_pages(
    tmp_path: Path, allocator_warnings: list[logging.LogRecord], value: str
):
    sysctl = tmp_path / "compact_unevictable_allowed"
    sysctl.write_text(value)
    warn_if_compaction_moves_mlocked_pages(str(sysctl))
    assert len(allocator_warnings) == 1
    assert "vm.compact_unevictable_allowed" in allocator_warnings[0].getMessage()


def test_sysctl_check_is_silent_when_compaction_skips_locked_pages(
    tmp_path: Path, allocator_warnings: list[logging.LogRecord]
):
    sysctl = tmp_path / "compact_unevictable_allowed"
    sysctl.write_text("0\n")
    warn_if_compaction_moves_mlocked_pages(str(sysctl))
    assert allocator_warnings == []


def test_sysctl_check_is_silent_without_the_sysctl(
    tmp_path: Path, allocator_warnings: list[logging.LogRecord]
):
    warn_if_compaction_moves_mlocked_pages(str(tmp_path / "missing"))
    assert allocator_warnings == []


@requires_pinned_memory
def test_mlock_allocator_warns_about_sysctl_only_when_enabled(
    allocator_warnings: list[logging.LogRecord],
):
    if Path("/proc/sys/vm/compact_unevictable_allowed").read_text().strip() == "0":
        pytest.skip("Node already sets vm.compact_unevictable_allowed=0")
    LazyMemoryAllocator(INIT_SIZE, INIT_SIZE, mlock=False).close()
    assert not any(
        "compact_unevictable_allowed" in r.getMessage() for r in allocator_warnings
    )
    LazyMemoryAllocator(INIT_SIZE, INIT_SIZE, mlock=True).close()
    assert any(
        "compact_unevictable_allowed" in r.getMessage() for r in allocator_warnings
    )
