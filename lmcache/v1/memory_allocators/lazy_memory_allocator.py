# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import List, Optional, Union
import ctypes
import mmap
import os
import threading

# Third Party
import torch

# First Party
from lmcache import device_ops, torch_dev, torch_device_type
from lmcache.logging import init_logger
from lmcache.v1.memory_allocators.tensor_memory_allocator import TensorMemoryAllocator
from lmcache.v1.memory_management import (
    AddressManager,
    MemoryAllocatorInterface,
    MemoryFormat,
    MemoryObj,
)
from lmcache.v1.platform import current_device_spec
from lmcache.v1.system_detection import NUMAMapping

logger = init_logger(__name__)

_libc = ctypes.CDLL(None, use_errno=True)
_libc.mlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
_libc.mlock.restype = ctypes.c_int
_libc.munlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
_libc.munlock.restype = ctypes.c_int
_libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
_libc.madvise.restype = ctypes.c_int

COMPACT_UNEVICTABLE_ALLOWED_PATH = "/proc/sys/vm/compact_unevictable_allowed"


# Helper functions
def warn_if_compaction_moves_mlocked_pages(
    path: str = COMPACT_UNEVICTABLE_ALLOWED_PATH,
) -> None:
    """
    Log a warning if memory compaction may still migrate mlocked pages.

    ``mlock`` keeps pages resident but does not stop compaction from moving
    them unless the ``vm.compact_unevictable_allowed`` sysctl is ``0``.

    Args:
        path (str): Path of the sysctl file to read. Defaults to
            ``/proc/sys/vm/compact_unevictable_allowed``.

    Note:
        Does nothing if the file cannot be read (non-Linux hosts, kernels
        without the sysctl).
    """
    try:
        with open(path) as f:
            value = f.read().strip()
    except OSError:
        return
    if value != "0":
        logger.warning(
            "L1 memory is mlocked, but vm.compact_unevictable_allowed=%s, so "
            "memory compaction can still migrate it. On ROCm each migrated "
            "page of registered host memory makes the driver evict this "
            "process's GPU queues and stalls transfers (LMCache #5361; with "
            "64 GB of mlocked L1 one forced compaction kept the queues "
            "evicted for 29 s per GPU, 5 s with the sysctl at 0). Set "
            "'sysctl -w vm.compact_unevictable_allowed=0' on the node.",
            value,
        )


def get_numa_id(numa_mapping: NUMAMapping) -> int:
    """
    Get the NUMA ID for the current GPU

    Args:
        numa_mapping (NUMAMapping): The NUMA mapping object.

    Returns:
        int: The NUMA ID for the current GPU.

    Raises:
        KeyError: If GPU id is not detected in the numa mapping.
    """
    gpu_id = torch_dev.current_device() if torch_dev.is_available() else 0
    return numa_mapping.gpu_to_numa_mapping[gpu_id]


def align_to(size: int, align_size: int) -> int:
    """
    Align the given size to the nearest multiple of align_size.

    Args:
        size (int): The size to align.
        align_size (int): The alignment size, MUST BE a power of two.

    Returns:
        int: The aligned size.
    """
    return (size + align_size - 1) & (~(align_size - 1))


# Main class
class LazyMemoryAllocator(MemoryAllocatorInterface):
    """
    Allocates CPU (numa) pinned memory with a initial size and expand
    the size to the required size in the background.

    Background expansion logic:
    - After registering X GB memory, we call sbrk and updates _curr_size
    - Once everything is registered, the background thread stops

    With ``mlock=True`` every chunk is mlocked before it is registered and
    the buffer is excluded from transparent huge pages. On ROCm, registered
    host memory is not page-locked, and every page that the kernel migrates
    (memory compaction, khugepaged collapsing pages into a huge page) makes
    the driver evict the process's GPU queues on all GPUs (LMCache #5361).
    """

    PIN_CHUNK_SIZE = 1 << 26  # 64 MB pin chunk
    COMMIT_SIZE = 1 << 30  # Do a commit every 1 GB
    LOG_INTERVAL = 10 << 30  # Log expansion progress every 10 GB

    def __init__(
        self,
        init_size: int,
        final_size: int,
        align_bytes: int = AddressManager.ALIGN_BYTES,
        numa_mapping: NUMAMapping | None = None,
        mlock: bool = False,
    ) -> None:
        """
        Args:
            init_size (int): Initial size of the memory allocation in bytes.
            final_size (int): Final size of the memory allocation in bytes.
            align_bytes (int, optional): Alignment for the underlying allocations.
                Must be a positive power of two. The buffer's base address is
                aligned to this value, not merely the offsets within it.
            numa_mapping (NUMAMapping | None, optional): If set, allocate the
                buffer on the NUMA node of the current GPU.
            mlock (bool, optional): ``mlock()`` every chunk right before it is
                pinned (initial chunk and expansion chunks), so that memory
                compaction cannot migrate it once
                ``vm.compact_unevictable_allowed`` is ``0``; logs a warning if
                that sysctl is not ``0``. Also ``madvise(MADV_NOHUGEPAGE)``
                the whole buffer before pinning, because khugepaged migrates
                mlocked pages when it collapses them into huge pages. A failed
                ``mlock`` (``RLIMIT_MEMLOCK`` too small, no ``CAP_IPC_LOCK``)
                is logged once and counted, a failed ``madvise`` is logged; the
                memory is still pinned and usable. ``close()`` unlocks the
                memory again.

        Raises:
            ValueError: If ``align_bytes`` is not a positive power of two.
            RuntimeError: If the platform does not support memory pinning, or if
                the allocated buffer could not be aligned to ``align_bytes``.
        """
        if align_bytes <= 0 or align_bytes & (align_bytes - 1) != 0:
            raise ValueError("align_bytes must be a positive power of two")

        # Whether using NUMA allocation
        self._use_numa = numa_mapping is not None
        # Currently pinned size, only accessed by the expansion thread
        self._curr_size = align_to(init_size, self.PIN_CHUNK_SIZE)
        # Final size of the allocation, only accessed by the expansion thread
        self._final_size = align_to(final_size, self.PIN_CHUNK_SIZE)
        # Underlying buffer for the memory allocation
        self._buffer: torch.Tensor
        if not current_device_spec.is_pin_supported:
            raise RuntimeError(
                f"Backend '{torch_device_type}' does not support memory "
                "pinning. LazyMemoryAllocator requires pinned memory."
            )

        # List of (ptr, size) for pinned memory chunks
        self._pin_record: list[tuple[int, int]] = []

        # Whether to mlock each chunk before pinning it
        self._mlock = mlock
        # Bytes mlocked so far and number of chunks whose mlock failed. Only
        # accessed by __init__, then by the expansion thread, then by close()
        # after the thread has been joined.
        self._mlocked_bytes = 0
        self._mlock_failures = 0
        if mlock:
            warn_if_compaction_moves_mlocked_pages()

        # Detect numa mapping
        if numa_mapping is not None:
            numa_id = get_numa_id(numa_mapping)
            ptr = device_ops.alloc_numa_ptr(self._final_size, numa_id)
            arr_type = ctypes.c_uint8 * self._final_size
            buf = arr_type.from_address(ptr)
            self._buffer = torch.frombuffer(buf, dtype=torch.uint8)
        else:
            # torch.empty() only guarantees 64-byte alignment, but consumers of
            # get_l1_memory_desc() (O_DIRECT, RDMA/GDS) need the buffer base
            # itself aligned to align_bytes.
            backing = torch.empty(
                self._final_size + align_bytes - 1,
                dtype=torch.uint8,
                device="cpu",
                pin_memory=False,
            )
            offset = (-backing.data_ptr()) % align_bytes
            # Slice shares storage with `backing`; no separate reference needed.
            self._buffer = backing[offset : offset + self._final_size]

        # Fail loudly here rather than let a misaligned buffer surface as an
        # O_DIRECT EINVAL somewhere downstream.
        base_ptr = self._buffer.data_ptr()
        if base_ptr % align_bytes != 0:
            raise RuntimeError(
                f"LazyMemoryAllocator buffer base {base_ptr:#x} is not aligned "
                f"to align_bytes={align_bytes} (remainder "
                f"{base_ptr % align_bytes})."
            )

        if mlock:
            self._disable_transparent_huge_pages()

        # Pin the first `curr_size` bytes (aligned to the internal chunk size)
        self._pin_memory_chunk(0, self._curr_size)

        # Create the tensor memory allocator
        self._allocator = TensorMemoryAllocator(
            tensor=self._buffer,
            align_bytes=align_bytes,
            init_address_space=self._curr_size,
        )

        # Get the address manager
        # NOTE(ApostaC): this assumes the tensor memory allocator owns the address
        # manager, which creates extra coupling in the code.
        # NOTE(ApostaC): this also assumes that the behavior of the allocation is
        # completely determined by the address manager.
        self._address_manager = self._allocator.address_manager

        # Launch the background expansion thread
        self._stop_expand = threading.Event()
        self._expand_thread = threading.Thread(
            target=self._expand_worker, daemon=True, name="lazy-mem-expand-thread"
        )
        self._expand_thread.start()

    # Public methods
    def allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        fmt: MemoryFormat = MemoryFormat.UNDEFINED,
        allocator_type: Optional[str] = None,
    ) -> Optional[MemoryObj]:
        """Allocate one object from the lazily pinned memory pool.

        Args:
            shapes: Logical tensor shape or shapes to allocate.
            dtypes: Logical tensor dtype or dtypes to allocate.
            fmt: Memory format stored in the returned metadata.
            allocator_type: Optional allocator type string.

        Returns:
            A memory object, or ``None`` if the committed address space is full.
        """
        obj = self._allocator.allocate(shapes, dtypes, fmt, allocator_type)
        # HACK(ApostaC): reset the parent allocator to this lazy allocator
        # There should be a cleaner way to decouple lazy allocator and
        # tensor memory allocator
        if obj is not None:
            obj.parent_allocator = self
        return obj

    def batched_allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.UNDEFINED,
        allocator_type: Optional[str] = None,
    ) -> Optional[List[MemoryObj]]:
        """Allocate a batch of objects from the lazily pinned memory pool.

        Args:
            shapes: Logical tensor shape or shapes to allocate for each object.
            dtypes: Logical tensor dtype or dtypes to allocate for each object.
            batch_size: Number of memory objects to allocate.
            fmt: Memory format stored in the returned metadata.
            allocator_type: Optional allocator type string.

        Returns:
            Memory objects for the batch, or ``None`` if allocation fails.
        """
        # HACK(ApostaC): reset the parent allocator to this lazy allocator
        # There should be a cleaner way to decouple lazy allocator and
        # tensor memory allocator
        ret = self._allocator.batched_allocate(
            shapes, dtypes, batch_size, fmt, allocator_type
        )

        if ret is None:
            return ret

        for obj in ret:
            obj.parent_allocator = self
        return ret

    def free(
        self,
        memory_obj: MemoryObj,
        allocator_type: Optional[str] = None,
    ) -> None:
        """Free one memory object back to the lazy allocator.

        Args:
            memory_obj: The memory object to free.
            allocator_type: Optional allocator type string.
        """
        self._allocator.free(memory_obj, allocator_type)

    def batched_free(
        self,
        memory_objs: List[MemoryObj],
        allocator_type: Optional[str] = None,
        update_stats: bool = True,
    ) -> None:
        """Free a batch of memory objects back to the lazy allocator.

        Args:
            memory_objs: Memory objects to free.
            allocator_type: Optional allocator type string.
            update_stats: Whether to update allocator statistics.
        """
        self._allocator.batched_free(memory_objs, allocator_type, update_stats)

    def close(self) -> None:
        """Stop background expansion and release pinned, mlocked or NUMA memory."""
        # Stop the background expansion thread
        self._stop_expand.set()
        self._expand_thread.join()

        # Unpin all pinned memory chunks
        for ptr, size in self._pin_record:
            current_device_spec.unpin_memory(ptr)
        self._pin_record.clear()

        if self._mlocked_bytes:
            # munlock ignores pages in the range that were never locked.
            _libc.munlock(self._buffer.data_ptr(), self._curr_size)
            self._mlocked_bytes = 0

        # Free the underlying buffer if using NUMA allocation
        if self._use_numa:
            device_ops.free_numa_ptr(self._buffer.data_ptr(), self._final_size)

    def memcheck(self) -> bool:
        """Return whether the delegated tensor allocator is consistent."""
        return self._allocator.memcheck()

    def get_underlying_buffer(self) -> torch.Tensor:
        """
        Get the underlying buffer tensor. Will be used by RDMA registrations.
        """
        return self._buffer

    def get_address_manager(self) -> AddressManager:
        """
        Get the address manager used by this allocator.
        """
        return self._address_manager

    # Helper functions
    def _pin_memory_chunk(self, offset: int, size: int) -> None:
        """
        Pin a chunk of memory.

        Args:
            offset (int): Offset in the buffer to pin.
            size (int): Size of the memory chunk in bytes.
        """
        assert offset & (self.PIN_CHUNK_SIZE - 1) == 0, (
            "Offset must be aligned to PIN_CHUNK_SIZE"
        )
        assert size & (self.PIN_CHUNK_SIZE - 1) == 0, (
            "Size must be aligned to PIN_CHUNK_SIZE"
        )
        assert offset + size <= self._final_size, "Pinning exceeds buffer size"

        ptr = self._buffer.data_ptr() + offset
        if self._mlock:
            self._mlock_chunk(ptr, size)
        # Use flag: cudaHostRegisterMapped (0x02)
        if not current_device_spec.pin_memory(ptr, size, 2):
            logger.warning(
                "pin_memory failed for chunk at ptr=%#x size=%d; "
                "DMA performance may be degraded",
                ptr,
                size,
            )
        else:
            self._pin_record.append((ptr, size))

    def _disable_transparent_huge_pages(self) -> None:
        """
        madvise(MADV_NOHUGEPAGE) the whole buffer. khugepaged collapses
        mlocked pages into huge pages too, which migrates them just like
        compaction does.
        """
        start = self._buffer.data_ptr() & ~(mmap.PAGESIZE - 1)
        end = align_to(self._buffer.data_ptr() + self._final_size, mmap.PAGESIZE)
        if _libc.madvise(start, end - start, mmap.MADV_NOHUGEPAGE) != 0:
            err = ctypes.get_errno()
            logger.warning(
                "LazyMemoryAllocator: madvise(MADV_NOHUGEPAGE) failed for L1 "
                "at ptr=%#x size=%d: %s (errno %d). With transparent huge "
                "pages enabled, khugepaged can still migrate L1 pages.",
                start,
                end - start,
                os.strerror(err),
                err,
            )

    def _mlock_chunk(self, ptr: int, size: int) -> None:
        """
        mlock one chunk. Logs the first failure only; later ones are counted
        and reported by ``_log_mlock_summary``.
        """
        if _libc.mlock(ptr, size) == 0:
            self._mlocked_bytes += size
            return
        err = ctypes.get_errno()
        self._mlock_failures += 1
        if self._mlock_failures == 1:
            logger.warning(
                "LazyMemoryAllocator: mlock failed for chunk at ptr=%#x "
                "size=%d: %s (errno %d), with %d bytes of L1 locked so far. "
                "Failed chunks are still pinned but stay movable; further "
                "failures are only counted. Give the server an unlimited "
                "memlock limit (e.g. docker run --ulimit memlock=-1) or "
                "CAP_IPC_LOCK, or pass --no-l1-mlock.",
                ptr,
                size,
                os.strerror(err),
                err,
                self._mlocked_bytes,
            )

    def _log_mlock_summary(self) -> None:
        """Log how much of the pinned L1 memory ended up mlocked."""
        if self._mlock_failures:
            logger.warning(
                "LazyMemoryAllocator: mlock failed for %d chunks; %d MB of "
                "%d MB L1 memory is mlocked.",
                self._mlock_failures,
                self._mlocked_bytes >> 20,
                self._curr_size >> 20,
            )
        else:
            logger.info(
                "LazyMemoryAllocator: mlocked %d MB of L1 memory.",
                self._mlocked_bytes >> 20,
            )

    def _commit_expansion(self, expand_size: int) -> None:
        """
        Call sbrk in the address manager to commit the expansion.
        """
        self._address_manager.sbrk(expand_size)

    def _log_expansion_progress(self, expanded_since_last_log: int) -> None:
        """
        Log the cumulative expansion progress since the last log.
        """
        percent = 100.0 * self._curr_size / self._final_size
        logger.info(
            "LazyMemoryAllocator: Expanded %s MB pinned memory, "
            "now total is %s MB / %s MB (%.1f%%)",
            expanded_since_last_log >> 20,
            self._curr_size >> 20,
            self._final_size >> 20,
            percent,
        )

    def _expand_worker(self) -> None:
        """
        Background worker to expand the pinned memory.
        """
        last_commit_size = self._curr_size
        last_log_size = self._curr_size
        while self._curr_size < self._final_size and not self._stop_expand.is_set():
            # Expand chunk by chunk and commit
            for i in range(self.COMMIT_SIZE // self.PIN_CHUNK_SIZE):
                if self._curr_size >= self._final_size:
                    break
                self._pin_memory_chunk(self._curr_size, self.PIN_CHUNK_SIZE)
                self._curr_size += self.PIN_CHUNK_SIZE

            expand_size = self._curr_size - last_commit_size
            self._commit_expansion(expand_size)
            last_commit_size = self._curr_size

            # Log every LOG_INTERVAL bytes, and always on the final commit.
            expanded_since_last_log = self._curr_size - last_log_size
            if (
                expanded_since_last_log >= self.LOG_INTERVAL
                or self._curr_size >= self._final_size
            ):
                self._log_expansion_progress(expanded_since_last_log)
                last_log_size = self._curr_size

        if self._mlock:
            self._log_mlock_summary()
