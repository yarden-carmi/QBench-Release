"""Per-op specifications for SuperPoint and SuperGlue.

Each ``OpSpec`` builds an ``OpInstance`` for a single op:

    instance = spec.build(generator)
    fp32_out   = instance.run_fp32()
    qbench_out = instance.run_qbench(q_type='fp8_e4m3')   # may raise on CPU
    ipu_in     = instance.ipu_inputs()                    # tensors handed to bridge

The same module instances are reused for FP32 and QBench (only `q_type`
differs) so weights are identical across the two paths.

Note on the simulator path: ``QuantConv2d`` / ``QuantConv1d`` /
``QuantSoftmax`` etc. require a CUDA tensor in float32 (the codec is
CUDA-only).  Ops without weighted layers (NMS, Threshold, Concat,
Sinkhorn, MatchSelect, ...) run on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Spec / instance dataclasses
# ---------------------------------------------------------------------------


@dataclass
class OpInstance:
    """Realised op for a single seed: the same weights drive fp32 and qbench."""

    name: str
    inputs: dict
    input_desc: str
    run_fp32: Callable[[], Any]
    run_qbench: Callable[[str], Any]
    ipu_inputs: Callable[[], dict] = field(default=lambda: {})


@dataclass
class OpSpec:
    name: str
    family: str           # 'superpoint' | 'superglue'
    layer_kind: str       # 'conv2d', 'conv1d', 'softmax', 'matmul', 'maxpool', ...
    build: Callable[[torch.Generator, torch.device], OpInstance]
    ipu_op_id: Optional[str] = None   # None = no IPU kernel intended yet


# ---------------------------------------------------------------------------
# Helpers shared across specs
# ---------------------------------------------------------------------------


def _randn(shape, g, device):
    return torch.randn(*shape, generator=g, device=device)


def _copy_state(src: nn.Module, dst: nn.Module) -> None:
    dst.load_state_dict(src.state_dict(), strict=True)


def _quantconv2d(src: nn.Conv2d, q_type: str) -> nn.Module:
    from runspace.src.ops.quant_conv import QuantConv2d
    qc = QuantConv2d(
        src.in_channels, src.out_channels, src.kernel_size,
        stride=src.stride, padding=src.padding, dilation=src.dilation,
        groups=src.groups, bias=src.bias is not None,
        q_type=q_type,
    )
    _copy_state(src, qc)
    qc.calibrate_weights()
    return qc


def _quantconv1d(src: nn.Conv1d, q_type: str) -> nn.Module:
    from runspace.src.ops.quant_conv1d import QuantConv1d
    qc = QuantConv1d(
        src.in_channels, src.out_channels, src.kernel_size,
        stride=src.stride, padding=src.padding, dilation=src.dilation,
        groups=src.groups, bias=src.bias is not None,
        q_type=q_type,
    )
    _copy_state(src, qc)
    qc.calibrate_weights()
    return qc


def _shape(t):
    if isinstance(t, torch.Tensor):
        return tuple(t.shape)
    if isinstance(t, (list, tuple)):
        return [_shape(x) for x in t]
    return type(t).__name__


# ---------------------------------------------------------------------------
# SuperPoint conv specs
# ---------------------------------------------------------------------------


def _make_conv2d_spec(name, in_ch, out_ch, k, padding, in_shape):
    def build(g, device):
        x = _randn(in_shape, g, device)
        ref = nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=padding).to(device)
        with torch.no_grad():
            ref.weight.copy_(_randn(ref.weight.shape, g, device) * 0.05)
            if ref.bias is not None:
                ref.bias.zero_()
        ref.eval()

        def run_fp32():
            with torch.no_grad():
                return ref(x)

        def run_qbench(q_type):
            qc = _quantconv2d(ref, q_type).to(device).eval()
            with torch.no_grad():
                return qc(x)

        return OpInstance(
            name=name,
            inputs={'x': x, 'weight': ref.weight, 'bias': ref.bias},
            input_desc=f"x={tuple(x.shape)}, w={tuple(ref.weight.shape)}",
            run_fp32=run_fp32,
            run_qbench=run_qbench,
            ipu_inputs=lambda: {'x': x, 'weight': ref.weight, 'bias': ref.bias},
        )

    return OpSpec(name=name, family='superpoint', layer_kind='conv2d', build=build,
                  ipu_op_id=f'superpoint.{name}')


def _make_conv1d_spec(name, family, in_ch, out_ch, in_shape):
    def build(g, device):
        x = _randn(in_shape, g, device)
        ref = nn.Conv1d(in_ch, out_ch, kernel_size=1).to(device)
        with torch.no_grad():
            ref.weight.copy_(_randn(ref.weight.shape, g, device) * 0.05)
            if ref.bias is not None:
                ref.bias.zero_()
        ref.eval()

        def run_fp32():
            with torch.no_grad():
                return ref(x)

        def run_qbench(q_type):
            qc = _quantconv1d(ref, q_type).to(device).eval()
            with torch.no_grad():
                return qc(x)

        return OpInstance(
            name=name,
            inputs={'x': x, 'weight': ref.weight, 'bias': ref.bias},
            input_desc=f"x={tuple(x.shape)}, w={tuple(ref.weight.shape)}",
            run_fp32=run_fp32,
            run_qbench=run_qbench,
            ipu_inputs=lambda: {'x': x, 'weight': ref.weight, 'bias': ref.bias},
        )

    return OpSpec(name=name, family=family, layer_kind='conv1d', build=build,
                  ipu_op_id=f'{family}.{name}')


# ---------------------------------------------------------------------------
# Non-weighted SuperPoint ops
# ---------------------------------------------------------------------------


def _spec_relu():
    def build(g, device):
        x = _randn((1, 64, 32, 32), g, device)

        def run_fp32():
            return F.relu(x)

        def run_qbench(q_type):
            return F.relu(x)

        return OpInstance(
            name='relu',
            inputs={'x': x},
            input_desc=f"x={tuple(x.shape)}",
            run_fp32=run_fp32,
            run_qbench=run_qbench,
            ipu_inputs=lambda: {'x': x},
        )

    return OpSpec(name='relu', family='superpoint', layer_kind='activation', build=build,
                  ipu_op_id='superpoint.relu')


def _spec_maxpool2d():
    def build(g, device):
        x = _randn((1, 64, 32, 32), g, device)

        def run_fp32():
            return F.max_pool2d(x, kernel_size=2, stride=2)

        def run_qbench(q_type):
            return F.max_pool2d(x, kernel_size=2, stride=2)

        return OpInstance(
            name='maxpool2d',
            inputs={'x': x},
            input_desc=f"x={tuple(x.shape)}",
            run_fp32=run_fp32,
            run_qbench=run_qbench,
            ipu_inputs=lambda: {'x': x},
        )

    return OpSpec(name='maxpool2d', family='superpoint', layer_kind='maxpool', build=build,
                  ipu_op_id='superpoint.maxpool2d')


def _spec_softmax_det():
    def build(g, device):
        x = _randn((1, 65, 8, 8), g, device)

        def run_fp32():
            return F.softmax(x, dim=1)

        def run_qbench(q_type):
            from runspace.src.ops.quant_softmax import QuantSoftmax
            sm = QuantSoftmax(dim=1, q_type=q_type, quant_mode='tensor').to(device)
            sm.eval()
            with torch.no_grad():
                return sm(x)

        return OpInstance(
            name='softmax_det',
            inputs={'x': x},
            input_desc=f"x={tuple(x.shape)}",
            run_fp32=run_fp32,
            run_qbench=run_qbench,
            ipu_inputs=lambda: {'x': x},
        )

    return OpSpec(name='softmax_det', family='superpoint', layer_kind='softmax', build=build,
                  ipu_op_id='superpoint.softmax_det')


def _spec_discard_trash():
    def build(g, device):
        x = _randn((1, 65, 8, 8), g, device)
        from runspace.src.ops.observed_ops import ObservedDiscardTrash
        op = ObservedDiscardTrash().to(device).eval()

        def run_fp32():
            with torch.no_grad():
                return x[:, :-1]

        def run_qbench(q_type):
            with torch.no_grad():
                return op(x)

        return OpInstance(
            name='discard_trash',
            inputs={'x': x},
            input_desc=f"x={tuple(x.shape)}",
            run_fp32=run_fp32,
            run_qbench=run_qbench,
            ipu_inputs=lambda: {'x': x},
        )

    return OpSpec(name='discard_trash', family='superpoint', layer_kind='reshape', build=build,
                  ipu_op_id='superpoint.discard_trash')


def _spec_reorder_reshape():
    def build(g, device):
        x = _randn((1, 64, 8, 8), g, device)
        from runspace.src.ops.observed_ops import ObservedReorderReshape
        op = ObservedReorderReshape().to(device).eval()

        def run_fp32():
            b, _, h, w = x.shape
            y = x.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
            return y.permute(0, 1, 3, 2, 4).reshape(b, h * 8, w * 8)

        def run_qbench(q_type):
            with torch.no_grad():
                return op(x)

        return OpInstance(
            name='reorder_reshape',
            inputs={'x': x},
            input_desc=f"x={tuple(x.shape)}",
            run_fp32=run_fp32,
            run_qbench=run_qbench,
            ipu_inputs=lambda: {'x': x},
        )

    return OpSpec(name='reorder_reshape', family='superpoint', layer_kind='reshape', build=build,
                  ipu_op_id='superpoint.reorder_reshape')


def _spec_simple_nms():
    def build(g, device):
        x = _randn((1, 32, 32), g, device).abs()
        from runspace.src.ops.observed_ops import ObservedSimpleNMS, _simple_nms
        op = ObservedSimpleNMS(radius=4).to(device).eval()

        def run_fp32():
            return _simple_nms(x, 4)

        def run_qbench(q_type):
            with torch.no_grad():
                return op(x)

        return OpInstance(
            name='simple_nms',
            inputs={'scores': x},
            input_desc=f"scores={tuple(x.shape)}, radius=4",
            run_fp32=run_fp32,
            run_qbench=run_qbench,
            ipu_inputs=lambda: {'scores': x, 'radius': 4},
        )

    return OpSpec(name='simple_nms', family='superpoint', layer_kind='maxpool', build=build,
                  ipu_op_id='superpoint.simple_nms')


def _spec_l2norm():
    def build(g, device):
        x = _randn((1, 256, 8, 8), g, device)
        from runspace.src.ops.observed_ops import ObservedL2Norm
        op = ObservedL2Norm().to(device).eval()

        def run_fp32():
            return F.normalize(x, p=2, dim=1)

        def run_qbench(q_type):
            with torch.no_grad():
                return op(x)

        return OpInstance(
            name='l2_norm',
            inputs={'x': x},
            input_desc=f"x={tuple(x.shape)}",
            run_fp32=run_fp32,
            run_qbench=run_qbench,
            ipu_inputs=lambda: {'x': x},
        )

    return OpSpec(name='l2_norm', family='superpoint', layer_kind='normalize', build=build,
                  ipu_op_id='superpoint.l2_norm')


def _spec_grid_sample():
    def build(g, device):
        descriptors = _randn((1, 256, 8, 8), g, device)
        kpts_norm = (torch.rand((1, 16, 2), generator=g, device=device) * 2 - 1)
        from runspace.src.ops.observed_ops import ObservedGridSample
        op = ObservedGridSample().to(device).eval()

        def run_fp32():
            b, c, _, _ = descriptors.shape
            desc = F.grid_sample(
                descriptors, kpts_norm.view(b, 1, -1, 2),
                mode='bilinear', align_corners=True,
            )
            return F.normalize(desc.reshape(b, c, -1), p=2, dim=1)

        def run_qbench(q_type):
            with torch.no_grad():
                return op(descriptors, kpts_norm)

        return OpInstance(
            name='grid_sample',
            inputs={'descriptors': descriptors, 'kpts_norm': kpts_norm},
            input_desc=f"desc={tuple(descriptors.shape)}, kpts={tuple(kpts_norm.shape)}",
            run_fp32=run_fp32,
            run_qbench=run_qbench,
            ipu_inputs=lambda: {'descriptors': descriptors, 'kpts_norm': kpts_norm},
        )

    return OpSpec(name='grid_sample', family='superpoint', layer_kind='grid_sample', build=build,
                  ipu_op_id='superpoint.grid_sample')


def _spec_coord_ops():
    def build(g, device):
        descriptors = _randn((1, 256, 8, 8), g, device)
        keypoints = (torch.rand((1, 16, 2), generator=g, device=device) * 64)
        from runspace.src.ops.observed_ops import ObservedCoordOps
        op = ObservedCoordOps(s=8).to(device).eval()

        def _ref():
            _, _, h, w = descriptors.shape
            s = 8
            kp = keypoints - s / 2 + 0.5
            denom = torch.tensor(
                [(w * s - s / 2 - 0.5), (h * s - s / 2 - 0.5)],
                dtype=kp.dtype, device=kp.device,
            )[None]
            kp = kp / denom
            return kp * 2 - 1

        return OpInstance(
            name='coord_ops',
            inputs={'descriptors': descriptors, 'keypoints': keypoints},
            input_desc=f"desc={tuple(descriptors.shape)}, kpts={tuple(keypoints.shape)}",
            run_fp32=_ref,
            run_qbench=lambda q_type: op(keypoints, descriptors),
            ipu_inputs=lambda: {'descriptors': descriptors, 'keypoints': keypoints, 's': 8},
        )

    return OpSpec(name='coord_ops', family='superpoint', layer_kind='coord', build=build,
                  ipu_op_id='superpoint.coord_ops')


def _spec_sample_descriptors():
    def build(g, device):
        descriptors = _randn((1, 256, 8, 8), g, device)
        keypoints = (torch.rand((1, 16, 2), generator=g, device=device) * 64)
        from runspace.src.ops.observed_ops import ObservedSampleDescriptors
        op = ObservedSampleDescriptors(s=8).to(device).eval()

        def run_fp32():
            _, _, h, w = descriptors.shape
            s = 8
            kp = keypoints - s / 2 + 0.5
            denom = torch.tensor(
                [(w * s - s / 2 - 0.5), (h * s - s / 2 - 0.5)],
                dtype=kp.dtype, device=kp.device,
            )[None]
            kp = (kp / denom) * 2 - 1
            b, c, _, _ = descriptors.shape
            d = F.grid_sample(descriptors, kp.view(b, 1, -1, 2),
                              mode='bilinear', align_corners=True)
            return F.normalize(d.reshape(b, c, -1), p=2, dim=1)

        return OpInstance(
            name='sample_descriptors',
            inputs={'descriptors': descriptors, 'keypoints': keypoints},
            input_desc=f"desc={tuple(descriptors.shape)}, kpts={tuple(keypoints.shape)}",
            run_fp32=run_fp32,
            run_qbench=lambda q_type: op(keypoints, descriptors),
            ipu_inputs=lambda: {'descriptors': descriptors, 'keypoints': keypoints, 's': 8},
        )

    return OpSpec(name='sample_descriptors', family='superpoint',
                  layer_kind='composed', build=build,
                  ipu_op_id='superpoint.sample_descriptors')


# ---------------------------------------------------------------------------
# SuperGlue ops
# ---------------------------------------------------------------------------


def _spec_keypoint_normalize():
    def build(g, device):
        N = 32
        kpts = (torch.rand((1, N, 2), generator=g, device=device) * 480)
        image_shape = (1, 1, 480, 640)
        from runspace.src.ops.observed_ops import ObservedKeypointNormalize
        op = ObservedKeypointNormalize().to(device).eval()

        def run_fp32():
            _, _, h, w = image_shape
            one = kpts.new_tensor(1)
            size = torch.stack([one * w, one * h])[None]
            center = size / 2
            scaling = size.max(1, keepdim=True).values * 0.7
            return (kpts - center[:, None, :]) / scaling[:, None, :]

        return OpInstance(
            name='keypoint_normalize',
            inputs={'kpts': kpts, 'image_shape': image_shape},
            input_desc=f"kpts={tuple(kpts.shape)}, image_shape={image_shape}",
            run_fp32=run_fp32,
            run_qbench=lambda q_type: op(kpts, image_shape),
            ipu_inputs=lambda: {'kpts': kpts, 'image_shape': image_shape},
        )

    return OpSpec(name='keypoint_normalize', family='superglue',
                  layer_kind='coord', build=build,
                  ipu_op_id='superglue.keypoint_normalize')


def _spec_attention_scores():
    def build(g, device):
        # Shape per SuperGlue MultiHeadedAttention: (B, dim/heads, heads, N)
        q = _randn((1, 64, 4, 32), g, device)
        k = _randn((1, 64, 4, 32), g, device)
        from runspace.src.ops.observed_ops import ObservedAttentionScores
        op = ObservedAttentionScores().to(device).eval()

        def run_fp32():
            return torch.einsum('bdhn,bdhm->bhnm', q, k) / q.shape[1] ** 0.5

        def run_qbench(q_type):
            op_q = ObservedAttentionScores(q_type=q_type).to(device).eval()
            with torch.no_grad():
                return op_q(q, k)

        return OpInstance(
            name='attention_scores',
            inputs={'query': q, 'key': k},
            input_desc=f"q={tuple(q.shape)}, k={tuple(k.shape)}",
            run_fp32=run_fp32,
            run_qbench=run_qbench,
            ipu_inputs=lambda: {'query': q, 'key': k},
        )

    return OpSpec(name='attention_scores', family='superglue',
                  layer_kind='matmul', build=build,
                  ipu_op_id='superglue.attention_scores')


def _spec_attention_apply():
    def build(g, device):
        prob = F.softmax(_randn((1, 4, 32, 32), g, device), dim=-1)
        v = _randn((1, 64, 4, 32), g, device)
        from runspace.src.ops.observed_ops import ObservedAttentionApply

        def run_fp32():
            return torch.einsum('bhnm,bdhm->bdhn', prob, v)

        def run_qbench(q_type):
            op_q = ObservedAttentionApply(q_type=q_type).to(device).eval()
            with torch.no_grad():
                return op_q(prob, v)

        return OpInstance(
            name='attention_apply',
            inputs={'prob': prob, 'value': v},
            input_desc=f"prob={tuple(prob.shape)}, v={tuple(v.shape)}",
            run_fp32=run_fp32,
            run_qbench=run_qbench,
            ipu_inputs=lambda: {'prob': prob, 'value': v},
        )

    return OpSpec(name='attention_apply', family='superglue',
                  layer_kind='matmul', build=build,
                  ipu_op_id='superglue.attention_apply')


def _spec_attention_softmax():
    def build(g, device):
        s = _randn((1, 4, 32, 32), g, device)

        def run_fp32():
            return F.softmax(s, dim=-1)

        def run_qbench(q_type):
            from runspace.src.ops.quant_softmax import QuantSoftmax
            sm = QuantSoftmax(dim=-1, q_type=q_type, quant_mode='tensor').to(device).eval()
            with torch.no_grad():
                return sm(s)

        return OpInstance(
            name='attention_softmax',
            inputs={'scores': s},
            input_desc=f"scores={tuple(s.shape)}",
            run_fp32=run_fp32,
            run_qbench=run_qbench,
            ipu_inputs=lambda: {'scores': s},
        )

    return OpSpec(name='attention_softmax', family='superglue',
                  layer_kind='softmax', build=build,
                  ipu_op_id='superglue.attention_softmax')


def _spec_desc_matmul():
    def build(g, device):
        d = 256
        desc0 = F.normalize(_randn((1, d, 32), g, device), p=2, dim=1)
        desc1 = F.normalize(_randn((1, d, 32), g, device), p=2, dim=1)
        from runspace.src.ops.observed_ops import ObservedDescMatmul

        def run_fp32():
            return torch.einsum('bdn,bdm->bnm', desc0, desc1) / d ** 0.5

        def run_qbench(q_type):
            op_q = ObservedDescMatmul(descriptor_dim=d, q_type=q_type).to(device).eval()
            with torch.no_grad():
                return op_q(desc0, desc1)

        return OpInstance(
            name='desc_matmul',
            inputs={'mdesc0': desc0, 'mdesc1': desc1},
            input_desc=f"desc0={tuple(desc0.shape)}, desc1={tuple(desc1.shape)}",
            run_fp32=run_fp32,
            run_qbench=run_qbench,
            ipu_inputs=lambda: {'mdesc0': desc0, 'mdesc1': desc1},
        )

    return OpSpec(name='desc_matmul', family='superglue', layer_kind='matmul', build=build,
                  ipu_op_id='superglue.desc_matmul')


def _spec_sinkhorn():
    def build(g, device):
        scores = _randn((1, 32, 32), g, device)
        alpha = torch.tensor(1.0, device=device)
        from runspace.src.ops.observed_ops import ObservedSinkhorn, _log_sinkhorn_iterations
        op = ObservedSinkhorn(iters=20).to(device).eval()

        def _ref():
            b, m, n = scores.shape
            one = scores.new_tensor(1)
            ms, ns = (m * one).to(scores), (n * one).to(scores)
            bins0 = alpha.expand(b, m, 1)
            bins1 = alpha.expand(b, 1, n)
            alpha_e = alpha.expand(b, 1, 1)
            couplings = torch.cat([torch.cat([scores, bins0], -1),
                                   torch.cat([bins1, alpha_e], -1)], 1)
            norm = -(ms + ns).log()
            log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
            log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
            log_mu = log_mu[None].expand(b, -1)
            log_nu = log_nu[None].expand(b, -1)
            Z = _log_sinkhorn_iterations(couplings, log_mu, log_nu, 20)
            return Z - norm

        return OpInstance(
            name='sinkhorn',
            inputs={'scores': scores, 'alpha': alpha},
            input_desc=f"scores={tuple(scores.shape)}, iters=20",
            run_fp32=_ref,
            run_qbench=lambda q_type: op(scores, alpha),
            ipu_inputs=lambda: {'scores': scores, 'alpha': alpha, 'iters': 20},
        )

    return OpSpec(name='sinkhorn', family='superglue', layer_kind='sinkhorn', build=build,
                  ipu_op_id='superglue.sinkhorn')


def _spec_match_select():
    def build(g, device):
        # Score matrix already includes dustbins along last row/col.
        s = _randn((1, 33, 33), g, device)
        from runspace.src.ops.observed_ops import ObservedMatchSelect, _arange_like
        op = ObservedMatchSelect(match_threshold=0.2).to(device).eval()

        def _ref():
            max0 = s[:, :-1, :-1].max(2)
            max1 = s[:, :-1, :-1].max(1)
            indices0, indices1 = max0.indices, max1.indices
            mutual0 = _arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
            mutual1 = _arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
            zero = s.new_tensor(0)
            mscores0 = torch.where(mutual0, max0.values.exp(), zero)
            mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
            valid0 = mutual0 & (mscores0 > 0.2)
            valid1 = mutual1 & valid0.gather(1, indices1)
            indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
            indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))
            return indices0, indices1, mscores0, mscores1

        return OpInstance(
            name='match_select',
            inputs={'scores': s},
            input_desc=f"scores={tuple(s.shape)}",
            run_fp32=_ref,
            run_qbench=lambda q_type: op(s),
            ipu_inputs=lambda: {'scores': s, 'match_threshold': 0.2},
        )

    return OpSpec(name='match_select', family='superglue',
                  layer_kind='argmax', build=build,
                  ipu_op_id='superglue.match_select')


def _spec_observed_add():
    def build(g, device):
        a = _randn((1, 256, 32), g, device)
        b = _randn((1, 256, 32), g, device)
        from runspace.src.ops.observed_ops import ObservedAdd

        def run_fp32():
            return a + b

        def run_qbench(q_type):
            op = ObservedAdd(q_type=q_type).to(device).eval()
            with torch.no_grad():
                return op(a, b)

        return OpInstance(
            name='observed_add',
            inputs={'a': a, 'b': b},
            input_desc=f"a={tuple(a.shape)}, b={tuple(b.shape)}",
            run_fp32=run_fp32,
            run_qbench=run_qbench,
            ipu_inputs=lambda: {'a': a, 'b': b},
        )

    return OpSpec(name='observed_add', family='superglue', layer_kind='arith', build=build,
                  ipu_op_id='superglue.observed_add')


def _spec_observed_concat():
    def build(g, device):
        a = _randn((1, 64, 32), g, device)
        b = _randn((1, 192, 32), g, device)
        from runspace.src.ops.observed_ops import ObservedConcat

        def run_fp32():
            return torch.cat([a, b], dim=1)

        def run_qbench(q_type):
            op = ObservedConcat(dim=1, q_type=q_type).to(device).eval()
            with torch.no_grad():
                return op(a, b)

        return OpInstance(
            name='observed_concat',
            inputs={'a': a, 'b': b},
            input_desc=f"a={tuple(a.shape)}, b={tuple(b.shape)}",
            run_fp32=run_fp32,
            run_qbench=run_qbench,
            ipu_inputs=lambda: {'a': a, 'b': b, 'dim': 1},
        )

    return OpSpec(name='observed_concat', family='superglue',
                  layer_kind='reshape', build=build,
                  ipu_op_id='superglue.observed_concat')


# ---------------------------------------------------------------------------
# Master registry
# ---------------------------------------------------------------------------


def build_op_specs() -> list[OpSpec]:
    specs: list[OpSpec] = []

    # SuperPoint encoder convs (8 layers).  Spatial size kept small to keep
    # the harness fast; identical channel counts to the real model.
    sp_in = (1, 1, 32, 32)
    specs.append(_make_conv2d_spec('sp_conv1a', 1,   64,  3, 1, sp_in))
    specs.append(_make_conv2d_spec('sp_conv1b', 64,  64,  3, 1, (1, 64, 32, 32)))
    specs.append(_make_conv2d_spec('sp_conv2a', 64,  128, 3, 1, (1, 64, 16, 16)))
    specs.append(_make_conv2d_spec('sp_conv2b', 128, 128, 3, 1, (1, 128, 16, 16)))
    specs.append(_make_conv2d_spec('sp_conv3a', 128, 256, 3, 1, (1, 128, 8, 8)))
    specs.append(_make_conv2d_spec('sp_conv3b', 256, 256, 3, 1, (1, 256, 8, 8)))
    specs.append(_make_conv2d_spec('sp_conv4a', 256, 128, 3, 1, (1, 256, 4, 4)))
    specs.append(_make_conv2d_spec('sp_conv4b', 128, 128, 3, 1, (1, 128, 4, 4)))

    # SuperPoint detector head
    specs.append(_make_conv2d_spec('sp_convPa', 128, 256, 3, 1, (1, 128, 4, 4)))
    specs.append(_make_conv2d_spec('sp_convPb', 256, 65,  1, 0, (1, 256, 4, 4)))

    # SuperPoint descriptor head
    specs.append(_make_conv2d_spec('sp_convDa', 128, 256, 3, 1, (1, 128, 4, 4)))
    specs.append(_make_conv2d_spec('sp_convDb', 256, 256, 1, 0, (1, 256, 4, 4)))

    # SuperPoint helpers
    specs.append(_spec_relu())
    specs.append(_spec_maxpool2d())
    specs.append(_spec_softmax_det())
    specs.append(_spec_discard_trash())
    specs.append(_spec_reorder_reshape())
    specs.append(_spec_simple_nms())
    specs.append(_spec_l2norm())
    specs.append(_spec_coord_ops())
    specs.append(_spec_grid_sample())
    specs.append(_spec_sample_descriptors())

    # SuperGlue keypoint encoder MLP (Conv1d k=1)
    N = 32
    specs.append(_make_conv1d_spec('sg_kenc_3_32',   'superglue', 3,   32,  (1, 3, N)))
    specs.append(_make_conv1d_spec('sg_kenc_32_64',  'superglue', 32,  64,  (1, 32, N)))
    specs.append(_make_conv1d_spec('sg_kenc_64_128', 'superglue', 64,  128, (1, 64, N)))
    specs.append(_make_conv1d_spec('sg_kenc_128_256', 'superglue', 128, 256, (1, 128, N)))

    # SuperGlue attention projections + merge + MLP
    specs.append(_make_conv1d_spec('sg_attn_proj',   'superglue', 256, 256, (1, 256, N)))
    specs.append(_make_conv1d_spec('sg_attn_merge',  'superglue', 256, 256, (1, 256, N)))
    specs.append(_make_conv1d_spec('sg_mlp_512_512', 'superglue', 512, 512, (1, 512, N)))
    specs.append(_make_conv1d_spec('sg_mlp_512_256', 'superglue', 512, 256, (1, 512, N)))
    specs.append(_make_conv1d_spec('sg_final_proj',  'superglue', 256, 256, (1, 256, N)))

    # SuperGlue helpers
    specs.append(_spec_keypoint_normalize())
    specs.append(_spec_attention_scores())
    specs.append(_spec_attention_softmax())
    specs.append(_spec_attention_apply())
    specs.append(_spec_desc_matmul())
    specs.append(_spec_sinkhorn())
    specs.append(_spec_match_select())
    specs.append(_spec_observed_add())
    specs.append(_spec_observed_concat())

    return specs


OP_SPECS: list[OpSpec] = build_op_specs()
