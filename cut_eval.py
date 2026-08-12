# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""JIT-compile and wrap the batched binary cut-evaluation / FM-gains CUDA kernels."""

from __future__ import annotations

import os
import sys
from functools import lru_cache

import torch
from torch.utils.cpp_extension import load

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cut_eval_kernel.cu")


def _ensure_build_env():
    """Make ninja/nvcc discoverable for torch cpp_extension JIT builds."""
    candidates = [
        os.path.dirname(os.path.abspath(torch.__file__)),
        os.path.join(os.path.dirname(os.path.dirname(torch.__file__)), "bin"),
        "/usr/local/cuda-12.2/bin",
        "/usr/local/cuda/bin",
    ]
    env_bin = os.path.join(sys.prefix, "bin")
    path_parts = [env_bin] + candidates + [os.environ.get("PATH", "")]
    os.environ["PATH"] = os.pathsep.join(p for p in path_parts if p)
    if "CUDA_HOME" not in os.environ:
        for cuda_home in ("/usr/local/cuda-12.2", "/usr/local/cuda"):
            if os.path.isdir(cuda_home):
                os.environ["CUDA_HOME"] = cuda_home
                break


@lru_cache(maxsize=1)
def _module():
    _ensure_build_env()
    return load(
        name="hyspecpro_cut_eval_v6",
        sources=[_SRC],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
        verbose=False,
    )


def build_bipartite_csr(g, device):
    """Build CSR-by-net and CSR-by-cell (int32) from a DGL bipartite graph."""
    cells, nets = g.edges(etype="connect")
    cells = cells.to(device=device, dtype=torch.int32)
    nets = nets.to(device=device, dtype=torch.int32)

    # CSR by net: pins grouped by net id
    order_n = torch.argsort(nets)
    cells_by_net = cells[order_n].contiguous()
    nets_sorted = nets[order_n]
    net_cnt = g.num_nodes("net")
    deg = torch.bincount(nets_sorted, minlength=net_cnt).to(torch.int32)
    net_ptr = torch.empty(net_cnt + 1, dtype=torch.int32, device=device)
    net_ptr[0] = 0
    net_ptr[1:] = torch.cumsum(deg, 0).to(torch.int32)

    # CSR by cell: nets grouped by cell id (for atomic-free FM scatter)
    order_c = torch.argsort(cells)
    nets_by_cell = nets[order_c].contiguous()
    cells_sorted = cells[order_c]
    cell_cnt = g.num_nodes("cell")
    cell_deg = torch.bincount(cells_sorted, minlength=cell_cnt).to(torch.int32)
    cell_ptr = torch.empty(cell_cnt + 1, dtype=torch.int32, device=device)
    cell_ptr[0] = 0
    cell_ptr[1:] = torch.cumsum(cell_deg, 0).to(torch.int32)

    return (
        net_ptr.contiguous(),
        cells_by_net,
        deg.contiguous(),
        cell_ptr.contiguous(),
        nets_by_cell,
    )


# Back-compat alias used by older bench scripts
def build_net_csr(g, device):
    net_ptr, cell_idx, deg, _, _ = build_bipartite_csr(g, device)
    return net_ptr, cell_idx, deg


class CutEvaluator:
    """Stateful evaluator + FM gains: CSR built once; assignments stay uint8."""

    # Must match MAX_WORDS*32 / FM_MAX_B in cut_eval_kernel.cu
    MAX_B = 1024
    MAX_FM_B = 64
    # Nets above this degree use a cooperative block-per-net cut kernel.
    HIGH_DEG_THRESH = 128

    def __init__(self, g, device, th_l, th_u):
        self.device = device
        self.th_l = float(th_l)
        self.th_u = float(th_u)
        self.cell_cnt = g.num_nodes("cell")
        self.net_cnt = g.num_nodes("net")
        (
            self.net_ptr,
            self.cell_idx,
            self.deg,
            self.cell_ptr,
            self.net_idx,
        ) = build_bipartite_csr(g, device)
        high_mask = self.deg > self.HIGH_DEG_THRESH
        self.high_net_ids = torch.nonzero(high_mask, as_tuple=False).flatten().to(
            dtype=torch.int32
        ).contiguous()
        self.low_net_ids = torch.nonzero(~high_mask, as_tuple=False).flatten().to(
            dtype=torch.int32
        ).contiguous()
        # COO net ids aligned with cell_idx (CSR-by-net pin order) for
        # edge-parallel FM gains on highly skewed graphs (e.g. circuit5M).
        self.net_ids_coo = torch.repeat_interleave(
            torch.arange(self.net_cnt, device=device, dtype=torch.int64),
            self.deg.to(torch.int64),
        ).contiguous()
        self._skewed = bool(int(self.deg.max()) >= 100_000)
        _module()  # trigger JIT once

    def __call__(self, population):
        """
        population: [cell_cnt, B] integer/bool/uint8 {0,1} (CPU or CUDA)
        returns: score float tensor on CPU, shape [B]
        """
        assign = population.to(device=self.device, dtype=torch.uint8).contiguous()
        if assign.shape[0] != self.cell_cnt:
            raise ValueError(f"expected {self.cell_cnt} cells, got {assign.shape[0]}")
        B = assign.shape[1]
        if B <= self.MAX_B:
            score, _, _ = _module().evaluate_cut_cuda(
                self.net_ptr,
                self.cell_idx,
                assign,
                self.deg,
                self.low_net_ids,
                self.high_net_ids,
                self.th_l,
                self.th_u,
            )
            return score.cpu()

        scores = []
        for b0 in range(0, B, self.MAX_B):
            chunk = assign[:, b0 : b0 + self.MAX_B].contiguous()
            score, _, _ = _module().evaluate_cut_cuda(
                self.net_ptr,
                self.cell_idx,
                chunk,
                self.deg,
                self.low_net_ids,
                self.high_net_ids,
                self.th_l,
                self.th_u,
            )
            scores.append(score)
        return torch.cat(scores, dim=0).cpu()

    def _compute_gains_edge_parallel(self, assign):
        """Load-balanced FM gains via edge-parallel index_add (skew-friendly)."""
        B = assign.shape[1]
        # sum1[net, b] = #pins in part 1
        pin_assign = assign[self.cell_idx].to(torch.int32)  # [E, B]
        sum1 = torch.zeros(
            self.net_cnt, B, device=self.device, dtype=torch.int32
        )
        sum1.index_add_(0, self.net_ids_coo, pin_assign)
        deg = self.deg.unsqueeze(1)  # [net, 1]
        cut = (sum1 > 0) & (sum1 < deg)
        pos0 = (cut & (sum1 == (deg - 1))).to(torch.float32)
        neg0 = ((~cut) & (sum1 == 0)).to(torch.float32)
        pos1 = (cut & (sum1 == 1)).to(torch.float32)
        neg1 = ((~cut) & (sum1 == deg)).to(torch.float32)
        c0 = pos0 - neg0
        c1 = pos1 - neg1
        edge_c0 = c0[self.net_ids_coo]  # [E, B]
        edge_c1 = c1[self.net_ids_coo]
        gain0 = torch.zeros(self.cell_cnt, B, device=self.device, dtype=torch.float32)
        gain1 = torch.zeros(self.cell_cnt, B, device=self.device, dtype=torch.float32)
        gain0.index_add_(0, self.cell_idx.to(torch.int64), edge_c0)
        gain1.index_add_(0, self.cell_idx.to(torch.int64), edge_c1)
        return torch.where(assign == 0, gain0, gain1)

    def compute_gains(self, population):
        """
        Batched FM gains for 2-way partitions.
        population: [cell_cnt, B] {0,1}
        returns: float32 gains [cell_cnt, B] on device
        """
        assign = population.to(device=self.device, dtype=torch.uint8).contiguous()
        if assign.shape[0] != self.cell_cnt:
            raise ValueError(f"expected {self.cell_cnt} cells, got {assign.shape[0]}")
        # Extreme degree skew: edge-parallel torch path beats one-thread-per-net.
        if self._skewed:
            return self._compute_gains_edge_parallel(assign)

        B = assign.shape[1]
        if B <= self.MAX_FM_B:
            return _module().compute_fm_gains_cuda(
                self.net_ptr,
                self.cell_idx,
                self.cell_ptr,
                self.net_idx,
                assign,
                self.deg,
                self.low_net_ids,
                self.high_net_ids,
            )

        # Chunk wide batches (should be rare for FM topK).
        gains = torch.empty(self.cell_cnt, B, device=self.device, dtype=torch.float32)
        for b0 in range(0, B, self.MAX_FM_B):
            chunk = assign[:, b0 : b0 + self.MAX_FM_B].contiguous()
            gains[:, b0 : b0 + chunk.shape[1]] = _module().compute_fm_gains_cuda(
                self.net_ptr,
                self.cell_idx,
                self.cell_ptr,
                self.net_idx,
                chunk,
                self.deg,
                self.low_net_ids,
                self.high_net_ids,
            )
        return gains
