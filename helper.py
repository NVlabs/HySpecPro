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
from torch.cuda.amp import GradScaler
import dgl.function as fn
from itertools import chain
import dgl

def read_hgr(hgr_file):
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
    #print("Num of cells: ", cells.shape, len(node_weights))
    #graph = dgl.heterograph({("cell", "connect", "net"): (cells, nets)})
    data_dict = {("cell", "connect", "net"): (cells, nets)}
    num_nodes_dict = {"cell": num_nodes, "net": num_edges}
    graph = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)
    #print(graph)
    graph.nodes["cell"].data["weight"] = torch.tensor(node_weights).to(torch.float)
    graph.nodes["net"].data["weight"] = torch.tensor(edge_weights).to(torch.float)
    return graph