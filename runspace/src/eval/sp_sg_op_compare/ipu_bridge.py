"""Cross-repo bridge to the IPU-emulator's per-op runners.

Today the IPU emulator only ships a fully-connected kernel.  Most
SuperPoint and SuperGlue ops therefore have no IPU implementation yet:
the bridge surfaces that as ``IpuKernelMissing`` so the comparison
report makes the gap explicit op-by-op.

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
    """The IPU has no kernel for this op yet."""


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
        try:
            return self._mod.run_op(op_id, inputs, dtype=dtype)
        except NotImplementedError as e:
            raise IpuKernelMissing(str(e) or 'kernel pending')
