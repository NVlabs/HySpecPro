# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from itertools import chain
import dgl
import numpy as np
from numba import njit


def read_hgr(hgr_file):
    """Original pure-Python hMetis/.hgr reader (kept for reference / fallback)."""
    with open(hgr_file, "r") as f:
        # read first line
        line = f.readline().split()
        num_edges, num_nodes = int(line[0]), int(line[1])
        edges = []
        edge_weights = [0] * num_edges
        node_weights = [1] * num_nodes
        if len(line) > 2:
            weight_type = int(line[2])
        else:
            weight_type = 0
        # read rest of file
        for i, line in enumerate(f):
            line = line.split()
            if i < num_edges:
                evec = list(map(int, line))
                if weight_type in [0, 10]:
                    edge_weights[i] = 1
                    edges.append(evec)
                else:
                    edge_weights[i] = evec[0]
                    edges.append(evec[1:])
            else:
                node_weights[i - num_edges] = int(line[0])
    assert len(edges) == num_edges
    edge_degrees = [len(e) for e in edges]
    cells = torch.tensor(list(chain.from_iterable(edges))) - 1
    nets = torch.arange(num_edges).repeat_interleave(torch.tensor(edge_degrees))
    data_dict = {("cell", "connect", "net"): (cells, nets)}
    num_nodes_dict = {"cell": num_nodes, "net": num_edges}
    graph = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)
    graph.nodes["cell"].data["weight"] = torch.tensor(node_weights).to(torch.float)
    graph.nodes["net"].data["weight"] = torch.tensor(edge_weights).to(torch.float)
    return graph


@njit(cache=True)
def _parse_hgr_body(data, num_edges, num_nodes, weight_type):
    """Two-pass ASCII integer scan of an hMetis body (bytes after the header newline).

    Returns 0-based cell/net pin lists plus weight arrays.
    """
    n = len(data)
    has_ew = (weight_type == 1) or (weight_type == 11)
    has_nw = (weight_type == 10) or (weight_type == 11)

    degrees = np.zeros(num_edges, dtype=np.int64)
    edge_weights = np.ones(num_edges, dtype=np.float32)

    # Pass 1: count pins / capture edge weights
    i = 0
    edge = 0
    while edge < num_edges and i < n:
        while i < n and (data[i] == 32 or data[i] == 9 or data[i] == 13 or data[i] == 10):
            i += 1
        if i >= n:
            break
        first = True
        deg = 0
        while i < n and data[i] != 10:
            while i < n and (data[i] == 32 or data[i] == 9 or data[i] == 13):
                i += 1
            if i >= n or data[i] == 10:
                break
            sign = 1
            if data[i] == 45:
                sign = -1
                i += 1
            val = 0
            any_digit = False
            while i < n and data[i] >= 48 and data[i] <= 57:
                any_digit = True
                val = val * 10 + (data[i] - 48)
                i += 1
            if not any_digit:
                i += 1
                continue
            val *= sign
            if has_ew and first:
                edge_weights[edge] = np.float32(val)
                first = False
            else:
                deg += 1
                first = False
        degrees[edge] = deg
        edge += 1
        if i < n and data[i] == 10:
            i += 1

    total_pins = 0
    for e in range(num_edges):
        total_pins += degrees[e]

    cells = np.empty(total_pins, dtype=np.int64)
    nets = np.empty(total_pins, dtype=np.int64)

    # Pass 2: fill (cell, net) pins
    i = 0
    edge = 0
    pin = 0
    while edge < num_edges and i < n:
        while i < n and (data[i] == 32 or data[i] == 9 or data[i] == 13 or data[i] == 10):
            i += 1
        if i >= n:
            break
        first = True
        while i < n and data[i] != 10:
            while i < n and (data[i] == 32 or data[i] == 9 or data[i] == 13):
                i += 1
            if i >= n or data[i] == 10:
                break
            sign = 1
            if data[i] == 45:
                sign = -1
                i += 1
            val = 0
            any_digit = False
            while i < n and data[i] >= 48 and data[i] <= 57:
                any_digit = True
                val = val * 10 + (data[i] - 48)
                i += 1
            if not any_digit:
                i += 1
                continue
            val *= sign
            if has_ew and first:
                first = False
            else:
                cells[pin] = val - 1
                nets[pin] = edge
                pin += 1
                first = False
        edge += 1
        if i < n and data[i] == 10:
            i += 1

    node_weights = np.ones(num_nodes, dtype=np.float32)
    if has_nw:
        node = 0
        while node < num_nodes and i < n:
            while i < n and (data[i] == 32 or data[i] == 9 or data[i] == 13 or data[i] == 10):
                i += 1
            if i >= n:
                break
            sign = 1
            if data[i] == 45:
                sign = -1
                i += 1
            val = 0
            any_digit = False
            while i < n and data[i] >= 48 and data[i] <= 57:
                any_digit = True
                val = val * 10 + (data[i] - 48)
                i += 1
            if any_digit:
                node_weights[node] = np.float32(val * sign)
                node += 1
            while i < n and data[i] != 10:
                i += 1
            if i < n and data[i] == 10:
                i += 1

    return cells, nets, edge_weights, node_weights, total_pins


def read_hgr_fast(hgr_file):
    """Fast hMetis/.hgr reader: whole-file mmap-style load + Numba integer scan.

    Produces the same DGL heterograph as ``read_hgr`` (cell/net bipartite graph
    with unit weights when the file is unweighted).
    """
    with open(hgr_file, "rb") as f:
        raw = f.read()
    nl = raw.find(b"\n")
    if nl < 0:
        raise ValueError(f"empty/invalid hgr file: {hgr_file}")
    header = raw[:nl].split()
    num_edges = int(header[0])
    num_nodes = int(header[1])
    weight_type = int(header[2]) if len(header) > 2 else 0

    body = np.frombuffer(raw, dtype=np.uint8, offset=nl + 1)
    cells, nets, edge_weights, node_weights, n_pins = _parse_hgr_body(
        body, num_edges, num_nodes, weight_type
    )
    if n_pins != cells.shape[0]:
        cells = cells[:n_pins]
        nets = nets[:n_pins]

    cells_t = torch.from_numpy(np.ascontiguousarray(cells))
    nets_t = torch.from_numpy(np.ascontiguousarray(nets))
    data_dict = {("cell", "connect", "net"): (cells_t, nets_t)}
    num_nodes_dict = {"cell": num_nodes, "net": num_edges}
    graph = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)
    # copy() so tensors own their storage (frombuffer views are read-only)
    graph.nodes["cell"].data["weight"] = torch.from_numpy(node_weights.copy())
    graph.nodes["net"].data["weight"] = torch.from_numpy(edge_weights.copy())
    return graph
