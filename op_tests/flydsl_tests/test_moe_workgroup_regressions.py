# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Numerical checks for XCD workgroup coverage."""

import pytest
import torch

from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage1, flydsl_moe_stage2

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx() != "gfx950",
    reason="gfx950 FlyDSL required",
)


def _routing(tokens, experts, topk, block_m=32):
    ids, eids = [], []
    for expert in range(experts):
        routes = [
            token | (slot << 24)
            for token in range(tokens)
            for slot in range(topk)
            if (token + slot) % experts == expert
        ]
        blocks = (len(routes) + block_m - 1) // block_m
        ids.extend(routes + [tokens] * (blocks * block_m - len(routes)))
        eids.extend([expert] * blocks)
    return dict(
        sorted_token_ids=torch.tensor(ids, dtype=torch.int32, device="cuda"),
        sorted_expert_ids=torch.tensor(eids, dtype=torch.int32, device="cuda"),
        num_valid_ids=torch.tensor(
            [len(ids), tokens], dtype=torch.int32, device="cuda"
        ),
        topk=topk,
    )


def _fp4(shape):
    return torch.randint(0, 256, shape, dtype=torch.uint8, device="cuda").view(
        dtypes.fp4x2
    )


def _scales(rows, columns, exponent=127):
    # A constant exponent is valid in both the sorted and preshuffled layouts.
    return torch.full(
        ((rows + 255) // 256 * 256, (columns + 255) // 256 * 8),
        exponent,
        dtype=torch.uint8,
        device="cuda",
    )


@pytest.mark.parametrize("stage", [1, 2])
@pytest.mark.parametrize("tokens", [1, 65, 97])
def test_xcd_swizzle_covers_every_tile(stage, tokens):
    torch.manual_seed(93)
    model_dim, inter_dim = 512, 384
    routing = _routing(tokens, experts=1, topk=1)
    sorted_rows = routing["sorted_token_ids"].numel()
    # Stage 1 has 6/18/24 CTAs; stage 2 has 2/6/8. Include a grid smaller
    # than eight, a remainder grid, and a divisible-by-eight control.
    if stage == 1:
        a = (torch.randn(tokens, model_dim, device="cuda") / 8).to(dtypes.fp8)
        weight = _fp4((1, 2 * inter_dim, model_dim // 2))
        a_scale = _scales(sorted_rows, model_dim)
        w_scale = _scales(2 * inter_dim, model_dim, 123)

        def run(swizzle):
            return flydsl_moe_stage1(
                a,
                weight,
                **routing,
                out=torch.full(
                    (tokens, 1, inter_dim), -17, dtype=torch.bfloat16, device="cuda"
                ),
                tile_m=32,
                tile_n=128,
                tile_k=256,
                gate_mode="interleave",
                a1_scale=a_scale,
                w1_scale=w_scale,
                persist_m=1,
                xcd_swizzle=swizzle,
            )

    else:
        a = (torch.randn(tokens, 1, inter_dim, device="cuda") / 8).to(dtypes.fp8)
        weight = _fp4((1, model_dim, inter_dim // 2))
        a_scale = _scales(sorted_rows, inter_dim)
        w_scale = _scales(model_dim, inter_dim, 123)
        weights = torch.ones(sorted_rows, dtype=torch.float32, device="cuda")

        def run(swizzle):
            return flydsl_moe_stage2(
                a,
                weight,
                **routing,
                tile_m=32,
                tile_n=256,
                tile_k=128,
                a2_scale=a_scale,
                w2_scale=w_scale,
                sorted_weights=weights,
                persist=False,
                xcd_swizzle=swizzle,
            )

    expected = run(0)
    actual = run(4)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.01)
