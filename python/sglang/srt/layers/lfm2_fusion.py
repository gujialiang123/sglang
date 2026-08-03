"""Fused Triton kernels for the LFM2.5 (Lfm2Moe) model.

Four kernels closing data-movement gaps found by an operator-level audit of
LFM2.5-8B-A1B on H200. All of them are pure data movement -- no new math is
introduced, and every kernel is validated bit-exact (or to a stated bound)
against the stock PyTorch sequence it replaces.

Short-convolution glue (``fused_gate_transpose`` / ``fused_transpose_gate``)
---------------------------------------------------------------------------
``Lfm2MoeShortConv.forward`` surrounds ``causal_conv1d_fn`` with three pure
data-movement operations::

    proj, _ = self.in_proj(hidden_states)        # [T, 3H]
    B_gate, C_gate, x = proj.chunk(3, dim=-1)    # strided views
    Bx = B_gate * x                              # elementwise  -> [T, H]
    Bx_t = Bx.transpose(0, 1).contiguous()       # materialise  -> [H, T]
    conv_out = causal_conv1d_fn(Bx_t, ...).transpose(0, 1)   # view, [T, H]
    output, _ = self.out_proj(C_gate * conv_out) # elementwise, reads transposed

``causal_conv1d_fn`` is an opaque external CUDA op requiring a ``[dim, seqlen]``
tensor with ``stride(-1) == 1``, so it acts as a barrier: the layout change
cannot be avoided, only *absorbed* into the neighbouring elementwise work.

Measured on long prefill these glue kernels move ~8.8 GB in 10.3 ms, i.e.
~0.83 TB/s on a part with ~4.8 TB/s of HBM bandwidth -- about **17 % of peak**.
That is the defect: ``Bx.transpose(0,1).contiguous()`` and the transposed read
inside ``C_gate * conv_out`` are uncoalesced. Folding the chunk, the gating
multiply and the transpose into one tiled kernel per side gives two compounding
wins: fewer passes over HBM, and each remaining pass running coalesced via a
register/shared-memory transpose instead of a strided per-element copy.

Isolated, at T=16000: 5.93x (input side) and 4.33x (output side), 0.98 ->
3.46 TB/s. Bit-exact at every tested shape.

These kernels sit on a ~30 us floor (Triton's Python launch path), so they only
pay off once there is enough work to amortise it -- measured crossover is
T ~= 2048. The caller guards on ``CONV_FUSION_MIN_TOKENS``. Decode never
transposes at all (``causal_conv1d_update`` consumes ``[T, H]`` directly), so
this pair is prefill-only by construction.

Decode gate multiply (``fused_gate_mul``)
-----------------------------------------
On decode, ``B_gate * x`` reads two *strided rows* of ``proj``. The access is
coalesced, but the strided rows prevent ``TensorIterator`` from vectorising:
the trace shows the scalar ``elementwise_kernel`` rather than
``vectorized_elementwise_kernel<8>``, leaving roughly half the bandwidth of an
equivalent contiguous multiply on the table. One Triton kernel reading ``proj``
directly avoids that without changing the launch count.

MoE reduction (``fused_moesum_add_rmsnorm``)
--------------------------------------------
The stock path materialises the reduced MoE output before the next layer
consumes it::

    partials[T, 4, H] -> moe_sum[T, H] -> fused_add_rmsnorm

Both steps are row-wise, so the intermediate is a wasted HBM round trip. This
kernel keeps the top-k reduction in registers, adds the residual, computes the
row RMS, and writes only the updated residual and the normalized activation.

Isolated: 2.46x at T=1, 2.68x at T=8, 1.30x at T=16000, but 0.72-0.74x in the
T=128..1024 range where the CUDA reducer wins. The caller therefore gates on
``T <= 32 or T >= MOESUM_FUSION_MIN_TOKENS``. Note this shape dependence is the
*opposite* of the short-conv kernels: those need large T to amortise launch
overhead, this one saves launch overhead plus a round trip and so wins most at
small T. Together they cover the whole range.

The residual output is bit-exact at every tested shape; the normalized output
is bit-exact through T=4096 and differs by 4.9e-4 at T=16000.

Tile shapes throughout come from a measured sweep (32 configurations per shape,
correctness-gated before timing), not from guesswork.
"""

from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl

# LFM2.5 routes to 4 of 32 experts.
TOP_K = 4


# ---------------------------------------------------------------------------
# Short conv, input side: chunk + gating multiply + transpose, in one pass.
#   reads  proj[:, 0:H] (B_gate) and proj[:, 2H:3H] (x)
#   writes out[H, T] = (B_gate * x)^T
# The tile is read coalesced along H and written coalesced along T; Triton keeps
# the transpose in registers/shared memory rather than issuing a strided global
# access per element.
# ---------------------------------------------------------------------------
@triton.jit
def _fused_gate_transpose_kernel(
    proj_ptr,
    out_ptr,
    T,
    H,
    stride_proj_t,
    stride_proj_h,
    stride_out_h,
    stride_out_t,
    BLOCK_T: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_t = offs_t < T
    mask_h = offs_h < H
    mask = mask_t[:, None] & mask_h[None, :]

    # [BLOCK_T, BLOCK_H] tile, contiguous along H
    base = offs_t[:, None] * stride_proj_t + offs_h[None, :] * stride_proj_h
    b = tl.load(proj_ptr + base, mask=mask, other=0.0)
    x = tl.load(proj_ptr + base + 2 * H * stride_proj_h, mask=mask, other=0.0)

    bx = (b.to(tl.float32) * x.to(tl.float32)).to(b.dtype)

    # write transposed: [BLOCK_H, BLOCK_T], contiguous along T
    out_off = offs_h[:, None] * stride_out_h + offs_t[None, :] * stride_out_t
    tl.store(out_ptr + out_off, tl.trans(bx), mask=mask_h[:, None] & mask_t[None, :])


def fused_gate_transpose(proj: torch.Tensor, H: int) -> torch.Tensor:
    """(B_gate * x)^T as one kernel. ``proj`` is [T, 3H]; returns [H, T]."""
    T = proj.shape[0]
    out = torch.empty((H, T), device=proj.device, dtype=proj.dtype)
    grid = lambda meta: (  # noqa: E731
        triton.cdiv(T, meta["BLOCK_T"]),
        triton.cdiv(H, meta["BLOCK_H"]),
    )
    _fused_gate_transpose_kernel[grid](
        proj,
        out,
        T,
        H,
        proj.stride(0),
        proj.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_T=64,
        BLOCK_H=128,
        num_warps=8,
        num_stages=2,
    )
    return out


# ---------------------------------------------------------------------------
# Short conv, output side: transpose + gating multiply, in one pass.
#   reads  conv_out[H, T] and proj[:, H:2H] (C_gate)
#   writes out[T, H] = C_gate * conv_out^T
# ---------------------------------------------------------------------------
@triton.jit
def _fused_transpose_gate_kernel(
    conv_ptr,
    proj_ptr,
    out_ptr,
    T,
    H,
    stride_conv_h,
    stride_conv_t,
    stride_proj_t,
    stride_proj_h,
    stride_out_t,
    stride_out_h,
    BLOCK_T: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_t = offs_t < T
    mask_h = offs_h < H

    # read conv_out as [BLOCK_H, BLOCK_T] (coalesced along T), then transpose
    conv_off = offs_h[:, None] * stride_conv_h + offs_t[None, :] * stride_conv_t
    c = tl.load(conv_ptr + conv_off, mask=mask_h[:, None] & mask_t[None, :], other=0.0)
    c_t = tl.trans(c)  # [BLOCK_T, BLOCK_H]

    # C_gate lives at column offset H, read coalesced along H
    mask = mask_t[:, None] & mask_h[None, :]
    g_off = offs_t[:, None] * stride_proj_t + (offs_h[None, :] + H) * stride_proj_h
    g = tl.load(proj_ptr + g_off, mask=mask, other=0.0)

    res = (g.to(tl.float32) * c_t.to(tl.float32)).to(g.dtype)
    out_off = offs_t[:, None] * stride_out_t + offs_h[None, :] * stride_out_h
    tl.store(out_ptr + out_off, res, mask=mask)


def fused_transpose_gate(
    conv_out: torch.Tensor, proj: torch.Tensor, H: int
) -> torch.Tensor:
    """C_gate * conv_out^T as one kernel. ``conv_out`` is [H, T]; returns [T, H]."""
    T = conv_out.shape[1]
    out = torch.empty((T, H), device=conv_out.device, dtype=conv_out.dtype)
    grid = lambda meta: (  # noqa: E731
        triton.cdiv(T, meta["BLOCK_T"]),
        triton.cdiv(H, meta["BLOCK_H"]),
    )
    _fused_transpose_gate_kernel[grid](
        conv_out,
        proj,
        out,
        T,
        H,
        conv_out.stride(0),
        conv_out.stride(1),
        proj.stride(0),
        proj.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_T=128,
        BLOCK_H=128,
        num_warps=8,
        num_stages=3,
    )
    return out


# ---------------------------------------------------------------------------
# Decode side: T is the batch size (1..32), nothing is bandwidth bound and
# nothing is transposed. The only cost is the strided-row read defeating
# vectorisation; one kernel that computes B*x and leaves C_gate to the caller
# avoids it.
# ---------------------------------------------------------------------------
@triton.jit
def _gate_mul_kernel(
    proj_ptr,
    out_ptr,
    T,
    H,
    stride_proj_t,
    stride_proj_h,
    stride_out_t,
    stride_out_h,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    n = T * H
    mask = offs < n
    t = offs // H
    h = offs % H
    base = t * stride_proj_t + h * stride_proj_h
    b = tl.load(proj_ptr + base, mask=mask, other=0.0)
    x = tl.load(proj_ptr + base + 2 * H * stride_proj_h, mask=mask, other=0.0)
    r = (b.to(tl.float32) * x.to(tl.float32)).to(b.dtype)
    tl.store(out_ptr + t * stride_out_t + h * stride_out_h, r, mask=mask)


def fused_gate_mul(proj: torch.Tensor, H: int) -> torch.Tensor:
    """B_gate * x without materialising the chunk views. Returns [T, H]."""
    T = proj.shape[0]
    out = torch.empty((T, H), device=proj.device, dtype=proj.dtype)
    n = T * H
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)  # noqa: E731
    _gate_mul_kernel[grid](
        proj,
        out,
        T,
        H,
        proj.stride(0),
        proj.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK=1024,
        num_warps=4,
    )
    return out


# ---------------------------------------------------------------------------
# MoE top-k reduction fused with the residual add and the following RMSNorm.
# ---------------------------------------------------------------------------
@triton.jit
def _moesum_add_rmsnorm_kernel(
    partials_ptr,
    residual_ptr,
    weight_ptr,
    output_ptr,
    H,
    stride_partials_t,
    stride_partials_k,
    stride_partials_h,
    stride_residual_t,
    stride_residual_h,
    stride_output_t,
    stride_output_h,
    eps,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    offs_h = tl.arange(0, BLOCK_H)
    mask = offs_h < H
    partial_base = row * stride_partials_t + offs_h * stride_partials_h

    summed = tl.zeros((BLOCK_H,), dtype=tl.float32)
    for k in tl.static_range(0, 4):
        value = tl.load(
            partials_ptr + partial_base + k * stride_partials_k,
            mask=mask,
            other=0.0,
        )
        summed += value.to(tl.float32)

    # The stock MoE reducer writes BF16 before fused_add_rmsnorm reads it;
    # round-tripping here keeps the fused kernel numerically aligned with it.
    summed = summed.to(tl.bfloat16).to(tl.float32)
    residual_offsets = row * stride_residual_t + offs_h * stride_residual_h
    residual = tl.load(residual_ptr + residual_offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    x = summed + residual

    tl.store(residual_ptr + residual_offsets, x, mask=mask)
    variance = tl.sum(x * x, axis=0) / H
    inv_rms = tl.rsqrt(variance + eps)
    weight = tl.load(weight_ptr + offs_h, mask=mask, other=0.0).to(tl.float32)
    output_offsets = row * stride_output_t + offs_h * stride_output_h
    tl.store(output_ptr + output_offsets, x * inv_rms * weight, mask=mask)


def fused_moesum_add_rmsnorm(
    partials: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reduce ``partials`` and apply residual-add RMSNorm in one kernel.

    ``partials`` is the un-combined FusedMoE output ``[T, TOP_K, H]``. Returns
    ``(normalized, residual)`` with ``residual`` updated in place, matching the
    calling convention of ``RMSNorm.forward_cuda(x, residual)``.
    """
    if partials.ndim != 3 or partials.shape[1] != TOP_K:
        raise ValueError(
            f"expected partials [T,{TOP_K},H], got {tuple(partials.shape)}"
        )
    T, _, H = partials.shape
    if residual.shape != (T, H):
        raise ValueError(f"expected residual {(T, H)}, got {tuple(residual.shape)}")
    if weight.shape != (H,):
        raise ValueError(f"expected weight {(H,)}, got {tuple(weight.shape)}")
    if (
        partials.dtype != torch.bfloat16
        or residual.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
    ):
        raise TypeError("moesum fusion currently supports BF16 only")
    if not partials.is_contiguous() or not residual.is_contiguous():
        raise ValueError("partials and residual must be contiguous")

    output = torch.empty_like(residual)
    block_h = triton.next_power_of_2(H)
    _moesum_add_rmsnorm_kernel[(T,)](
        partials,
        residual,
        weight,
        output,
        H,
        partials.stride(0),
        partials.stride(1),
        partials.stride(2),
        residual.stride(0),
        residual.stride(1),
        output.stride(0),
        output.stride(1),
        eps,
        BLOCK_H=block_h,
        num_warps=8,
        num_stages=1,
    )
    return output, residual
