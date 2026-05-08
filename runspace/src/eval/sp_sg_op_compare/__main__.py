"""CLI for the SuperPoint/SuperGlue per-op comparison.

Examples:

    python -m runspace.src.eval.sp_sg_op_compare \
        --dtype fp8_e4m3 \
        --output runspace/outputs/sp_sg_compare.csv

    # Subset by family or by name
    python -m runspace.src.eval.sp_sg_op_compare --family superpoint
    python -m runspace.src.eval.sp_sg_op_compare --only sp_conv1a,sg_kenc_3_32

    # Point at the IPU emulator checkout (otherwise: kernel_not_implemented)
    IPU_EMULATOR_REPO=/path/to/ipu-emulator \
        python -m runspace.src.eval.sp_sg_op_compare
"""

from __future__ import annotations

import argparse
import sys

import torch

from runspace.src.eval.sp_sg_op_compare.ipu_bridge import IpuBridge
from runspace.src.eval.sp_sg_op_compare.runner import (
    format_table,
    run_all,
    write_csv,
)


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dtype', default='fp8_e4m3',
                   help="QBench quant dtype (e.g. fp8_e4m3, fp8_e5m2).")
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default=None,
                   help='cpu | cuda | cuda:0 (default: auto)')
    p.add_argument('--only', default=None,
                   help='Comma-separated list of op names to run.')
    p.add_argument('--family', default=None, choices=(None, 'superpoint', 'superglue'),
                   help='Restrict to a single family.')
    p.add_argument('--ipu-repo', default=None,
                   help='Path to the ipu-emulator checkout '
                        '(overrides IPU_EMULATOR_REPO).')
    p.add_argument('--output', default=None,
                   help='Optional CSV output path.')
    p.add_argument('--quiet', action='store_true',
                   help='Suppress per-op progress lines.')
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)

    device = torch.device(args.device) if args.device else None
    only = [s.strip() for s in args.only.split(',')] if args.only else None
    bridge = IpuBridge(ipu_repo=args.ipu_repo)

    if not bridge.available:
        sys.stderr.write(
            "[sp_sg_op_compare] IPU emulator not importable: "
            f"{bridge.import_error}\n"
            "[sp_sg_op_compare] Continuing - every op will report "
            "ipu_status=ipu_unavailable.\n"
        )

    def _progress(spec):
        if not args.quiet:
            sys.stderr.write(f"[{spec.family}] {spec.name} ...\n")
            sys.stderr.flush()

    results = run_all(
        dtype=args.dtype, seed=args.seed, device=device,
        only=only, family=args.family,
        bridge=bridge, progress=_progress,
    )

    print(format_table(results))

    if args.output:
        write_csv(results, args.output)
        sys.stderr.write(f"[sp_sg_op_compare] wrote CSV: {args.output}\n")

    n_qb_ok = sum(1 for r in results if r.qbench_status == 'ok')
    n_ipu_ok = sum(1 for r in results if r.ipu_status == 'ok')
    n_ipu_missing = sum(1 for r in results if r.ipu_status == 'kernel_not_implemented')
    sys.stderr.write(
        f"[sp_sg_op_compare] summary: total={len(results)}, "
        f"qbench_ok={n_qb_ok}, ipu_ok={n_ipu_ok}, ipu_missing={n_ipu_missing}\n"
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
