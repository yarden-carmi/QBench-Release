"""Driver for the per-op SuperPoint/SuperGlue comparison."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Callable, Optional, Sequence

import torch

from runspace.src.eval.sp_sg_op_compare.op_specs import OP_SPECS, OpSpec
from runspace.src.eval.sp_sg_op_compare.ipu_bridge import (
    IpuBridge,
    IpuIsaGap,
    IpuKernelMissing,
    IpuUnavailable,
)


@dataclass
class CompareResult:
    name: str
    family: str
    layer_kind: str
    input_desc: str
    qbench_dtype: str
    qbench_status: str
    ipu_status: str
    qbench_vs_fp32_max_abs: Optional[float] = None
    qbench_vs_fp32_mean_abs: Optional[float] = None
    qbench_vs_fp32_cosine: Optional[float] = None
    ipu_vs_fp32_max_abs: Optional[float] = None
    ipu_vs_fp32_mean_abs: Optional[float] = None
    ipu_vs_fp32_cosine: Optional[float] = None
    ipu_vs_qbench_max_abs: Optional[float] = None
    ipu_vs_qbench_mean_abs: Optional[float] = None
    ipu_vs_qbench_cosine: Optional[float] = None
    notes: str = ''


def _flatten(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to('cpu').to(torch.float64).reshape(-1)


def _coerce_pair(a: Any, b: Any) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    """Return (a_flat, b_flat) when both are tensor-like and shape-compatible.

    Tuple/list outputs (e.g. match_select returns 4 tensors) are flattened
    by concatenating each entry, so a single set of metrics covers them.
    """
    def _to_flat(x):
        if isinstance(x, torch.Tensor):
            return _flatten(x)
        if isinstance(x, (list, tuple)):
            parts = [_to_flat(e) for e in x]
            parts = [p for p in parts if p is not None]
            return torch.cat(parts) if parts else None
        return None

    fa = _to_flat(a)
    fb = _to_flat(b)
    if fa is None or fb is None:
        return None
    if fa.numel() != fb.numel():
        return None
    return fa, fb


def _max_abs(a, b):
    p = _coerce_pair(a, b)
    if p is None:
        return None
    return float(torch.max(torch.abs(p[0] - p[1])))


def _mean_abs(a, b):
    p = _coerce_pair(a, b)
    if p is None:
        return None
    return float(torch.mean(torch.abs(p[0] - p[1])))


def _cosine(a, b):
    p = _coerce_pair(a, b)
    if p is None:
        return None
    fa, fb = p
    na = torch.linalg.norm(fa)
    nb = torch.linalg.norm(fb)
    if float(na) == 0.0 or float(nb) == 0.0:
        return float('nan')
    return float(torch.dot(fa, fb) / (na * nb))


def _summarise_notes(*items: str) -> str:
    return ' | '.join(s for s in items if s)


def run_one(
    spec: OpSpec,
    bridge: IpuBridge,
    *,
    dtype: str,
    seed: int,
    device: torch.device,
) -> CompareResult:
    g = torch.Generator(device='cpu').manual_seed(seed)
    instance = spec.build(g, device)

    fp32_out = None
    qbench_out = None
    qbench_status = 'pending'
    ipu_status = 'pending'
    notes_parts: list[str] = []

    try:
        fp32_out = instance.run_fp32()
    except Exception as e:
        notes_parts.append(f"fp32_error: {type(e).__name__}: {e}")

    try:
        qbench_out = instance.run_qbench(dtype)
        qbench_status = 'ok'
    except Exception as e:
        qbench_status = 'error'
        notes_parts.append(f"qbench_error: {type(e).__name__}: {e}")

    res = CompareResult(
        name=spec.name,
        family=spec.family,
        layer_kind=spec.layer_kind,
        input_desc=instance.input_desc,
        qbench_dtype=dtype,
        qbench_status=qbench_status,
        ipu_status=ipu_status,
    )

    if qbench_status == 'ok' and fp32_out is not None:
        res.qbench_vs_fp32_max_abs = _max_abs(qbench_out, fp32_out)
        res.qbench_vs_fp32_mean_abs = _mean_abs(qbench_out, fp32_out)
        res.qbench_vs_fp32_cosine = _cosine(qbench_out, fp32_out)

    if not bridge.available:
        res.ipu_status = 'ipu_unavailable'
        notes_parts.append(f"ipu_import: {bridge.import_error}")
    elif spec.ipu_op_id is None:
        res.ipu_status = 'kernel_not_planned'
    else:
        try:
            ipu_out = bridge.run(spec.ipu_op_id, instance.ipu_inputs(), dtype=dtype)
            res.ipu_status = 'ok'
            if fp32_out is not None:
                res.ipu_vs_fp32_max_abs = _max_abs(ipu_out, fp32_out)
                res.ipu_vs_fp32_mean_abs = _mean_abs(ipu_out, fp32_out)
                res.ipu_vs_fp32_cosine = _cosine(ipu_out, fp32_out)
            if qbench_out is not None:
                res.ipu_vs_qbench_max_abs = _max_abs(ipu_out, qbench_out)
                res.ipu_vs_qbench_mean_abs = _mean_abs(ipu_out, qbench_out)
                res.ipu_vs_qbench_cosine = _cosine(ipu_out, qbench_out)
        except IpuIsaGap as e:
            res.ipu_status = 'isa_gap'
            notes_parts.append(f"ipu: {e}")
        except IpuKernelMissing as e:
            res.ipu_status = 'kernel_not_implemented'
            notes_parts.append(f"ipu: {e}")
        except IpuUnavailable as e:
            res.ipu_status = 'ipu_unavailable'
            notes_parts.append(f"ipu: {e}")
        except Exception as e:
            res.ipu_status = 'error'
            notes_parts.append(f"ipu_error: {type(e).__name__}: {e}")

    res.notes = _summarise_notes(*notes_parts)
    return res


def run_all(
    *,
    dtype: str = 'fp8_e4m3',
    seed: int = 0,
    device: Optional[torch.device] = None,
    only: Optional[Sequence[str]] = None,
    family: Optional[str] = None,
    bridge: Optional[IpuBridge] = None,
    progress: Optional[Callable[[OpSpec], None]] = None,
) -> list[CompareResult]:
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if bridge is None:
        bridge = IpuBridge()
    only_set = set(only) if only else None

    results: list[CompareResult] = []
    for spec in OP_SPECS:
        if only_set and spec.name not in only_set:
            continue
        if family and spec.family != family:
            continue
        if progress is not None:
            progress(spec)
        res = run_one(spec, bridge, dtype=dtype, seed=seed, device=device)
        results.append(res)
    return results


def format_table(results: Sequence[CompareResult]) -> str:
    headers = (
        'name', 'family', 'kind', 'qbench', 'ipu',
        'qb_max', 'qb_mean', 'qb_cos',
        'ipu_max', 'ipu_mean', 'ipu_cos',
    )

    def _fmt(v):
        if v is None:
            return '-'
        if isinstance(v, float):
            return f'{v:.3e}'
        return str(v)

    rows = []
    for r in results:
        rows.append((
            r.name, r.family, r.layer_kind, r.qbench_status, r.ipu_status,
            _fmt(r.qbench_vs_fp32_max_abs), _fmt(r.qbench_vs_fp32_mean_abs),
            _fmt(r.qbench_vs_fp32_cosine),
            _fmt(r.ipu_vs_fp32_max_abs), _fmt(r.ipu_vs_fp32_mean_abs),
            _fmt(r.ipu_vs_fp32_cosine),
        ))

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    sep = '  '

    def _line(row):
        return sep.join(str(c).ljust(widths[i]) for i, c in enumerate(row))

    out = [_line(headers), _line(tuple('-' * w for w in widths))]
    for row in rows:
        out.append(_line(row))
    return '\n'.join(out)


def write_csv(results: Sequence[CompareResult], path: str) -> None:
    import csv
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not results:
        path.write_text('')
        return
    fields = list(asdict(results[0]).keys())
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))
