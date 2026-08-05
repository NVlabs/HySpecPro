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

import dgl
import dgl.function as fn
import torch
import numpy as np
from helper import read_hgr

import dgl.function as fn
import random
import os
import random, math, argparse
import cma

import cupy as cp
import torch
from torch.utils.dlpack import to_dlpack, from_dlpack
from cupyx.scipy.sparse import coo_matrix
from cupyx.scipy.sparse.linalg import eigsh

# batch size dictionary
design_dict = {
    "ibm01": 13000,
    "ibm02": 10000,
    "ibm03": 7000,
    "ibm04": 6000,
    "ibm05": 5000,
    "ibm06": 5000,
    "ibm07": 4000,
    "ibm08": 3500,
    "ibm09": 3500,
    "ibm10": 2600,
    "ibm11": 2200,
    "ibm12": 2000,
    "ibm13": 1500,
    "ibm14": 1200,
    "ibm15": 1000,
    "ibm16": 900,
    "ibm17": 900,
    "ibm18": 800,
    "sparcT1_core": 1860,
    "neuron": 1700,
    "stereo_vision": 1700,
    "des90": 1300,
    "SLAM_spheric": 1200,
    "cholesky_mc": 1300,
    "segmentation": 1060,
    "bitonic_mesh": 760,
    "dart": 760,
    "openCV": 660,
    "stap_qrd": 620,
    "minres": 600,
    "cholesky_bdti": 590,
    "denoise": 580,
    "sparcT2_core": 560,
    "gsm_switch": 420,
    "mes_noc": 380,
    "LU230": 380,
    "LU_Network": 360,
    "sparcT1_chip2": 300,
    "directrf": 240,
    "bitcoin_miner": 240
}
UB = 0.02
KWAY = 2
BIPARTITE_EMBED = True
design_root = "./benchmark/Titan23_benchmark/"
result_root = "./results/"

# build bipartite Laplacian from dgl graph
def bipartite_L_from_dgl(g, etype=('cell', 'connect', 'net'), device="cuda:0", symmetric=False):
    u_cell, v_net = g.edges(etype=etype)
    u_cell = u_cell.to(device)
    v_net = v_net.to(device)

    n_cells = g.num_nodes(etype[0])
    n_nets = g.num_nodes(etype[2])
    n_total = n_cells + n_nets
    net_offset = n_cells

    # === 1️⃣ Compute net degrees |e| (how many cells per net)
    net_deg = torch.zeros(n_nets, device=device, dtype=torch.float32)
    net_deg.scatter_add_(0, v_net, torch.ones_like(v_net, dtype=torch.float32))
    net_deg = net_deg.clamp(min=1.0)

    # === 2️⃣ Edge weights normalization
    if symmetric:
        # symmetric scaling: 1/sqrt(|e|)
        w_edge = 1.0 / torch.sqrt(net_deg[v_net])
    else:
        # Zhou normalization: 1/|e|
        w_edge = 1.0 / net_deg[v_net]

    # Duplicate edges for undirected bipartite graph
    src_all = torch.cat([u_cell, v_net + net_offset], dim=0).long()
    dst_all = torch.cat([v_net + net_offset, u_cell], dim=0).long()
    data_all = torch.cat([w_edge, w_edge], dim=0).float()

    # === 3️⃣ Convert to CuPy COO for GPU sparse ops
    src_cp = cp.fromDlpack(to_dlpack(src_all))
    dst_cp = cp.fromDlpack(to_dlpack(dst_all))
    data_cp = cp.fromDlpack(to_dlpack(data_all))

    A_cp = coo_matrix((data_cp, (src_cp, dst_cp)), shape=(n_total, n_total), dtype=cp.float32)
    A_cp.sum_duplicates()

    # === 4️⃣ Degree
    deg_cp = cp.asarray(A_cp.sum(axis=1)).ravel()
    deg_inv_sqrt = 1.0 / cp.sqrt(deg_cp + 1e-12)

    # === 5️⃣ Normalize adjacency
    row_idx = A_cp.row.astype(cp.int64)
    col_idx = A_cp.col.astype(cp.int64)
    data_scaled = A_cp.data * deg_inv_sqrt[row_idx] * deg_inv_sqrt[col_idx]
    A_norm = coo_matrix((data_scaled, (row_idx, col_idx)), shape=(n_total, n_total), dtype=cp.float32)
    A_norm.sum_duplicates()
    A_norm = 0.5 * (A_norm + A_norm.T)  # enforce symmetry

    # === 6️⃣ Laplacian
    eye_idx = cp.arange(n_total, dtype=cp.int64)
    I_cp = coo_matrix((cp.ones(n_total, dtype=cp.float32), (eye_idx, eye_idx)), shape=(n_total, n_total))
    L_cp = I_cp - A_norm

    # === 7️⃣ Sanity checks
    ones = cp.ones(n_total, dtype=cp.float32)
    print("L*1 max abs:", float(cp.max(cp.abs(L_cp @ ones))))

    sqrt_deg = cp.sqrt(deg_cp + 1e-12)
    vec = sqrt_deg  # D^{1/2} * 1
    res = L_cp @ vec
    print("L_sym * D^{1/2}1 max abs:", float(cp.max(cp.abs(res))))

    return L_cp

# compute k smallest eigenvectors of bipartite Laplacian using cupyx eigsh
def bipartite_spectral_embeddings_from_dgl(g, etype=('cell','connect','net'), k=16, which='SA', tol=1e-6, maxiter=5000, device="cuda:0"):
    L_cp = bipartite_L_from_dgl(g, etype=etype, device=device)

    eigvals_cp, eigvecs_cp = eigsh(L_cp, k=k, which=which, tol=tol, maxiter=maxiter)

    n_cells = g.num_nodes(etype[0])
    eigvecs_cells_cp = eigvecs_cp[:n_cells, :]  # (n_cells, k)

    # Convert eigvecs_cells_cp to torch tensor via DLPack (zero-copy)
    eigvecs_cells_torch = from_dlpack(eigvecs_cells_cp.toDlpack()).to(device=device)
    eigvals_cp = from_dlpack(eigvals_cp.toDlpack()).to(device=device)

    # Return cupy eigenvalues and torch embeddings for cells
    return eigvals_cp, eigvecs_cells_torch


# main function to run the partitioner
def run_partitioner(design, device, tag):

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    config = {
        "hgr_file": design_root + design + ".hgr",
        "hMetis_solution_head": "",
        "solution_file": result_root + tag + "_" + design + "_best_solution.pt",
        "score_file": result_root + tag + "_" + design + "_best_score.pt",
        "device": device,
        "use_weight": False,
        "N_batch": design_dict[design],
        "K": KWAY,
        "e": UB
    }

    hgr_file = config["hgr_file"]
    solution_file = config["solution_file"]
    use_weight = config["use_weight"]
    N_batch = config["N_batch"]
    K = config["K"]
    e = config["e"]

    g = read_hgr(hgr_file)
    cell_cnt, net_cnt = g.num_nodes("cell"), g.num_nodes("net")
    print(g)
    g = g.to(device)

    W = cell_cnt
    th_l = W*(1/K - e)
    th_u = W*(1/K + e)
    print("Partition size constraints: ", th_l, th_u)

    # batch evaluation function
    def evaluation(g, population):
        cell_cnt, B = population.shape
        with torch.no_grad():
            # Use float32 0/1 assignments for efficient DGL message passing
            assign_b = population.to(device=device, dtype=torch.float32)
            g.nodes["cell"].data["assign_b"] = assign_b

            # Aggregate number of part-1 pins per net per head
            g['connect'].update_all(fn.copy_u('assign_b', 'm'), fn.sum('m', 'net_sum1'), etype='connect')
            net_sum1 = g.nodes["net"].data["net_sum1"].to(torch.float32)

            # Prepare per-net degrees broadcast to [net_cnt, B] (cache by B)
            deg_b = g.nodes["net"].data.get("deg_b", None)
            if (deg_b is None) or (deg_b.shape[1] != B):
                deg = g.in_degrees(torch.arange(g.num_nodes('net'), device=device), etype='connect').to(torch.float32)
                deg_b = deg.view(-1, 1).repeat(1, B)
                g.nodes["net"].data["deg_b"] = deg_b

            # A net is cut if it has at least one pin in both parts
            net_cut = (net_sum1 > 0.0) & (net_sum1 < deg_b)
            N_cut = net_cut.sum(dim=0).to(torch.float32)

            # Balance penalty
            if use_weight:
                # Fallback to one-hot path only when weighted cells are enabled
                temp = torch.zeros([cell_cnt, B, K], device=device, dtype=torch.float32)
                temp.scatter_(2, population.unsqueeze(2).long(), 1)
                w = torch.sum(g.nodes["cell"].data["weight"].reshape(-1, 1, 1) * temp, 0)
                penalty = 50 * torch.sum(torch.relu(w - th_u) + torch.relu(th_l - w), 1)
            else:
                ones_per_batch = assign_b.sum(dim=0)
                zeros_per_batch = float(cell_cnt) - ones_per_batch
                weight_mat = torch.stack([zeros_per_batch, ones_per_batch], dim=1)  # [B,2]
                penalty = 50.0 * torch.sum(torch.relu(weight_mat - th_u) + torch.relu(th_l - weight_mat), dim=1)

            score = (N_cut + penalty).cpu()
            return score

    # conduct CMA-ES
    def CMA_ES_iterations(g, embed_list):
        best_solution_list = []
        best_score_list = []

        # conduct 5 runs of CMA-ES for each embedding
        for _ in range(5):
            for embed_idx, embed in enumerate(embed_list):
                print("embed_idx: ", embed_idx)
                # conduct CMA-ES
                with torch.no_grad():
                    N_nodes, N_embed = embed.shape
                    N_cma_ite = 50
                    best_solution = None
                    best_score = None
                    sigma = 0.3
                    Np = 3 # population size = N_batch * Np * Ns, where Np is the parallelization factor and Ns is the sequential factor
                    Ns = 1
                    Na = 1
                    es = cma.CMAEvolutionStrategy(np.zeros((N_embed)*2*Na), sigma, {'popsize': N_batch * Np * Ns})
                    unchange_cnt = 0
                    global_min = 100000000.0
                    RELU = torch.nn.ReLU().to(device)
                    for _ in range(N_cma_ite):
                        # stop if the best score has not improved for 3 consecutive iterations
                        if unchange_cnt >= 5:
                            break
                        # sample a new population
                        assign_W = es.ask()
                        assign_W_torch  = np.array(assign_W)
                        assign_W_torch  = torch.from_numpy(assign_W_torch).reshape(-1, Na, N_embed, 2).to(device).float() # N_batch * Np * Ns, N_embed, 1

                        # evaluate the population
                        temp_score_list = []
                        for s in range(Ns):
                            assign_W_torch_s = assign_W_torch[N_batch * Np * s : N_batch * Np * (s+1)]
                            logits = torch.einsum('ne,paek->pank', embed, assign_W_torch_s) # N_batch, Na, N_nodes, 2
                            # logits = RELU(logits)
                            logits = logits.sum(dim=1) # N_batch, N_nodes, 2

                            # logits = embed @ assign_W_torch_s  # N_batch, N_nodes, 1
                            logits = logits.permute(1, 0, 2)
                            # logits = torch.cat((logits,  - logits), 2) # N_nodes, N_batch, 2
                            solution = torch.argmax(logits, 2)
                            score = evaluation(g, solution)
                            temp_score_list.append(score.detach())

                            if best_score is None:
                                best_score = score.detach()
                                best_solution = solution.detach()
                            else:
                                mask = score < best_score
                                best_score[mask] = score[mask]
                                best_solution[:,mask] = solution[:,mask].detach()
                                del mask

                            del logits, solution, score, assign_W_torch_s

                        temp_score_list = torch.cat(temp_score_list, 0).cpu().tolist()
                        # print("temp_score_list: ", len(temp_score_list))
                        temp_min = min(temp_score_list)
                        if temp_min < global_min:
                            global_min = temp_min
                            unchange_cnt = 0
                        else:
                            unchange_cnt += 1

                        # update the CMA-ES with the population scores
                        es.tell(assign_W, temp_score_list)
                        es.disp()
                        del assign_W_torch, assign_W
                    del es
                    print("best score after CMA-ES: ", best_score.min())

                scores_sorted, indices = torch.sort(best_score)
                best_score = best_score[indices[:3]]
                best_solution = best_solution[:, indices[:3]]
                del indices, scores_sorted

                best_solution_list.append(best_solution)
                best_score_list.append(best_score)


        best_solution_all = torch.cat(best_solution_list, 1)
        best_score_all = torch.cat(best_score_list, 0)
        return best_solution_all, best_score_all

    # conduct batched FM/KL refinement
    def fm_kl_refine_batch(g, assign_batch, th_l, th_u, max_passes=2, max_iters=1000):
        """
        Batched FM/KL refinement.
        - assign_batch: LongTensor [cell_cnt, B] of {0,1}
        Returns: (refined_assign_batch [cell_cnt, B], improved_mask [B])
        """
        assign_batch = assign_batch.long()
        device_local = assign_batch.device
        cell_cnt = g.num_nodes('cell')
        B = assign_batch.shape[1]

        # Baseline scores per head
        with torch.no_grad():
            baseline_scores = evaluation(g, assign_batch)
        baseline_cut = baseline_scores.to(device_local).float()  # [B]

        part0_sizes = (assign_batch == 0).sum(dim=0).float()  # [B]
        part1_sizes = (cell_cnt - part0_sizes)

        improved_any = torch.zeros(B, dtype=torch.bool, device=device_local)

        def compute_gains_batched(assign_b):
            # Build one-hot per head: [cell_cnt, B, 2]
            onehot = torch.zeros(cell_cnt, B, 2, device=device_local, dtype=torch.float32)
            onehot.scatter_(2, assign_b.unsqueeze(2), 1.0)
            g.nodes['cell'].data['part_onehot_b'] = onehot
            g['connect'].update_all(fn.copy_u('part_onehot_b','m'), fn.sum('m','net_counts_b'), etype='connect')
            net_counts = g.nodes['net'].data['net_counts_b']  # [net_cnt, B, 2]
            net_cut = (net_counts[:,:,0] > 0) & (net_counts[:,:,1] > 0)  # [net_cnt, B]
            pos0 = (net_cut & (net_counts[:,:,0] == 1)).to(torch.float32)
            neg0 = ((~net_cut) & (net_counts[:,:,1] == 0)).to(torch.float32)
            pos1 = (net_cut & (net_counts[:,:,1] == 1)).to(torch.float32)
            neg1 = ((~net_cut) & (net_counts[:,:,0] == 0)).to(torch.float32)

            cells_idx, nets_idx = g.in_edges(torch.arange(g.num_nodes('net'), device=device_local), etype='connect')
            # edge contributions [E,B]
            edge_c_pos0 = pos0[nets_idx, :]
            edge_c_neg0 = neg0[nets_idx, :]
            edge_c_pos1 = pos1[nets_idx, :]
            edge_c_neg1 = neg1[nets_idx, :]

            gain0 = torch.zeros(cell_cnt, B, device=device_local, dtype=torch.float32)
            gain1 = torch.zeros(cell_cnt, B, device=device_local, dtype=torch.float32)
            gain0.index_add_(0, cells_idx, (edge_c_pos0 - edge_c_neg0))
            gain1.index_add_(0, cells_idx, (edge_c_pos1 - edge_c_neg1))
            gains = torch.where(assign_b == 0, gain0, gain1)
            return gains

        for _pass in range(max_passes):
            moved = torch.zeros(cell_cnt, B, dtype=torch.bool, device=device_local)
            iters = 0
            while iters < max_iters:
                gains = compute_gains_batched(assign_batch)
                gains_masked = gains.masked_fill(moved, float('-inf'))
                best_gain, best_idx = torch.max(gains_masked, dim=0)  # [B]
                try_mask = best_gain > 0
                if not try_mask.any():
                    break

                cols = torch.arange(B, device=device_local)
                cur_part = assign_batch[best_idx, cols]

                # Balance check per head
                new_p0 = torch.where(cur_part == 0, part0_sizes - 1, part0_sizes + 1)
                new_p1 = cell_cnt - new_p0
                balance_ok = (new_p0 >= th_l) & (new_p0 <= th_u) & (new_p1 >= th_l) & (new_p1 <= th_u)
                cand_mask = try_mask & balance_ok
                if not cand_mask.any():
                    # mark tried as moved and continue
                    moved[best_idx[try_mask], cols[try_mask]] = True
                    iters += 1
                    continue

                # Tentatively flip all candidate heads in a copy and evaluate in one shot
                tentative = assign_batch.clone()
                flip_cols = cols[cand_mask]
                flip_rows = best_idx[cand_mask]
                tentative[flip_rows, flip_cols] = 1 - cur_part[cand_mask]

                with torch.no_grad():
                    new_scores = evaluation(g, tentative).to(device_local).float()

                accept_mask = cand_mask & (new_scores < baseline_cut)
                if accept_mask.any():
                    a_cols = cols[accept_mask]
                    a_rows = best_idx[accept_mask]
                    assign_batch[a_rows, a_cols] = 1 - assign_batch[a_rows, a_cols]
                    # update baselines and sizes for accepted
                    improved_any[accept_mask] = True
                    delta = torch.where(cur_part[accept_mask] == 0, -1.0, 1.0)
                    part0_sizes[accept_mask] = part0_sizes[accept_mask] + delta
                    part1_sizes[accept_mask] = cell_cnt - part0_sizes[accept_mask]
                    baseline_cut[accept_mask] = new_scores[accept_mask]
                    moved[a_rows, a_cols] = True
                else:
                    # None accepted; mark tried as moved to avoid reselecting
                    moved[best_idx[cand_mask], cols[cand_mask]] = True

                iters += 1

        return assign_batch, improved_any

    # reduce graph by removing large nets
    print("Analyzing net degrees...")
    with torch.no_grad():
        D = g.in_degrees(torch.arange(net_cnt, device=device), etype='connect')
        max_degree = D.max().item()
        print(f"Max net degree: {max_degree}")

        top_percent_value = torch.quantile(D.float(), 0.9999)
        print(f"Top 0.01% net degree: {top_percent_value.item()}")

        degree_threshold = max(120, min(200, top_percent_value.item()))
        mask = D >= degree_threshold
        n_large_nets = mask.sum().item()
        print(f"Removing {n_large_nets} large nets (degree >= {degree_threshold})")

    if n_large_nets > 0:
        ori_g = g.to("cpu")
        ori_cell_cnt = ori_g.num_nodes('cell')
        remove_net_idx = torch.nonzero(mask, as_tuple=True)[0]
        edges = g.in_edges(remove_net_idx, etype='connect')
        remove_eids = g.edge_ids(edges[0], edges[1], etype="connect")
        g = dgl.remove_edges(g, remove_eids, "connect")
        g = dgl.remove_nodes(g, remove_net_idx, "net")
        assert g.num_nodes('cell') == ori_cell_cnt, \
            f"Cell count changed! Original: {ori_cell_cnt}, After: {g.num_nodes('cell')}"
        del mask, edges, remove_eids, D, remove_net_idx
        torch.cuda.empty_cache()
        new_g_flag = True
        print(f"Graph reduced: {ori_g.num_nodes('net')} -> {g.num_nodes('net')} nets")
        print(f"Cell count unchanged: {g.num_nodes('cell')} cells")
    else:
        # with torch.no_grad():
        #     D_after = g.in_degrees(torch.arange(g.num_nodes('net'), device=device), etype='connect').float()
        #     g.nodes["net"].data["deg"] = D_after.view(-1,1).repeat(1, N_batch)
        new_g_flag = False

    # construct embedding spaces
    embed_list = []

    if BIPARTITE_EMBED:
        # Compute spectral embedding from bipartite graph
        k = 48
        eigvals, eigvecs_cells_torch = bipartite_spectral_embeddings_from_dgl(g, etype=('cell','connect','net'), k=k, which='SA', device=device)
        print("Bipartite graph eigvals:", eigvals)
        mask = eigvals > 1e-6
        eigvals = eigvals[mask]
        eigvecs_cells_torch = eigvecs_cells_torch[:, mask]
        embed_list += [eigvecs_cells_torch[:, :4], eigvecs_cells_torch[:, :8], eigvecs_cells_torch[:, :16], eigvecs_cells_torch[:, :24], eigvecs_cells_torch[:, :36]]

    if new_g_flag:
        g = ori_g.to(device)
        new_g_flag = False

    best_solution, best_score = CMA_ES_iterations(g, embed_list)
    print("Best score after CMA-ES on all embedding spaces: ", best_score.min())

    if new_g_flag:
        print("Restoring original graph for final evaluation...")
        g = ori_g.to(device)
        best_score = evaluation(g, best_solution)
        scores_sorted, indices = torch.sort(best_score)
        best_solution = best_solution[:, indices]
        best_score = scores_sorted
        print(f"Final score on original graph: {best_score.min()}")
    else:
        scores_sorted, indices = torch.sort(best_score)
        best_solution = best_solution[:, indices]
        best_score = scores_sorted

    # FM/KL refinement on the best discrete solutions
    topK = 32
    best_assign = best_solution[:,:topK].to(device)
    refined_assign, did_improve = fm_kl_refine_batch(g, best_assign, th_l, th_u, max_passes=10)
    if did_improve.sum() > 0:
        print("FM/KL refinement improved the solution. Re-evaluating...")
        best_solution[:,:topK] = refined_assign.to(best_solution.device)
        best_score[:topK] = evaluation(g, best_solution[:,:topK])
        print(f"Refined final score: {best_score.min(), best_score[0]}")

        scores_sorted, indices = torch.sort(best_score)
        best_solution = best_solution[:, indices]

    torch.save(best_solution[:,0].to("cpu"), solution_file)
    return best_score.min().item()

def main():
    parser = argparse.ArgumentParser(description="HySpecPro")
    parser.add_argument("--design", type=str, default="sparcT1_core", help="Design name")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.add_argument("--tag", type=str, default="expected_cut_v2_fmrefine_v1", help="Tag to differentiate different runs")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    best_score = run_partitioner(args.design, args.device, args.tag)
    print(f"Final best score: {best_score}")

if __name__ == "__main__":
    main()