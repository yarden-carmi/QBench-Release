"""Per-op comparison harness for SuperPoint and SuperGlue.

For every op in the SuperPoint and SuperGlue forward graphs this package
constructs a paired execution path:

  * fp32:    a clean PyTorch reference using the same weights / inputs.
  * qbench:  the QBench simulator path (Observed* / Quant* op).
  * ipu:     the corresponding IPU-emulator kernel via the cross-repo
             bridge in ``ipu_bridge``.  Ops without a kernel are reported
             as ``kernel_not_implemented`` so the gap is visible.

Outputs per-op metrics (max_abs / mean_abs / cosine) and a CSV summary.
"""
