"""Cross-repo bridge to the IPU-emulator's per-op runners.

The bridge talks to ``ipu_apps.sp_sg_compare.run_op`` and translates
its three failure modes into distinct comparison-report statuses:

  * **ipu_status='ok'** — runner returned a tensor.
  * **ipu_status='kernel_not_implemented'** — runner raised
    :class:`NotImplementedError` (op has a known recipe but no
    asm/harness yet, or the kernel exists but the spec's input shape
    doesn't fit).  Reported via :class:`IpuKernelMissing`.
  * **ipu_status='isa_gap'** — runner raised the IPU side's ``IsaGap``
    sentinel (subclass of NotImplementedError).  The op cannot be
    expressed in the opcode-reference ISA at all (e.g. softmax /
    attention / sinkhorn dropped per fa16059).  Reported via
    :class:`IpuIsaGap`.

Resolution order for the IPU package:

  1. ``IPU_EMULATOR_REPO`` env var (path to the ipu-emulator checkout).
     The ipu_apps src dir is added to sys.path.
  2. Already-importable ``ipu_apps.sp_sg_compare`` (e.g. when the test
     environment installs ipu-apps).
  3. Otherwise: every op is reported as ``ipu_unavailable``.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Any, Optional


class IpuKernelMissing(RuntimeError):
    """The IPU has no kernel for this op yet (recipe is known)."""


class IpuIsaGap(RuntimeError):
    """The op is outside the opcode-reference ISA on this branch.

    Distinct from :class:`IpuKernelMissing`: missing-kernel ops have a
    known recipe and just need the asm + harness to be written; ISA-gap
    ops would require new ISA primitives, or were intentionally dropped
    (e.g. softmax / attention / sinkhorn per fa16059).
    """


class IpuUnavailable(RuntimeError):
    """The ipu-emulator package itself could not be located."""


_IPU_PKG = 'ipu_apps.sp_sg_compare'


def _add_ipu_repo_to_path(repo_path: Path) -> None:
    src = repo_path / 'src' / 'tools' / 'ipu-apps' / 'src'
    if not src.is_dir():
        raise IpuUnavailable(
            f"IPU repo at {repo_path} does not contain "
            f"src/tools/ipu-apps/src — is the path correct?"
        )
    sp = str(src.resolve())
    if sp not in sys.path:
        sys.path.insert(0, sp)


class IpuBridge:
    """Lazily imports the IPU per-op dispatcher.

    The bridge does not assume anything about the IPU side beyond a
    callable named ``run_op(op_id: str, inputs: dict, *, dtype: str)``
    that returns a ``torch.Tensor`` (or tuple) on success, or raises
    ``NotImplementedError`` for ops without a kernel.
    """

    def __init__(self, ipu_repo: Optional[str] = None) -> None:
        self._available = False
        self._import_error: Optional[str] = None
        self._mod = None

        repo = ipu_repo or os.environ.get('IPU_EMULATOR_REPO')
        try:
            if repo:
                _add_ipu_repo_to_path(Path(repo))
            self._mod = importlib.import_module(_IPU_PKG)
            self._available = True
        except Exception as e:
            self._import_error = f"{type(e).__name__}: {e}"

    @property
    def available(self) -> bool:
        return self._available

    @property
    def import_error(self) -> Optional[str]:
        return self._import_error

    def run(self, op_id: str, inputs: dict, *, dtype: str) -> Any:
        if not self._available:
            raise IpuUnavailable(self._import_error or 'ipu-emulator unavailable')
        # IsaGap is a subclass of NotImplementedError; check it first
        # by name so we don't have to import the IPU module here.
        IsaGap = getattr(self._mod, 'IsaGap', None)
        try:
            return self._mod.run_op(op_id, inputs, dtype=dtype)
        except NotImplementedError as e:
            if IsaGap is not None and isinstance(e, IsaGap):
                raise IpuIsaGap(str(e) or 'isa gap')
            raise IpuKernelMissing(str(e) or 'kernel pending')
