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
from helper import read_hgr_fast as read_hgr
from cut_eval import CutEvaluator

import os, argparse
import time
import cma
import cupy as cp
from torch.utils.dlpack import to_dlpack, from_dlpack
from cupyx.scipy.sparse import coo_matrix
from cupyx.scipy.sparse.linalg import eigsh

# batch size dictionary
batch_size_dict = {
    "sparcT1_core": 1000,
    "neuron": 1000,
    "stereo_vision": 1000,
    "des90": 1000,
    "SLAM_spheric": 1000,
    "cholesky_mc": 1000,
    "segmentation": 1000,
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
    "directrf": 300,
    "bitcoin_miner": 300,
    "Bump_2911.mtx": 180,
    "CurlCurl_4.mtx": 200,
    "Ga41As41H72.mtx": 200,
    "Geo_1438.mtx": 250,
    "HV15R.mtx": 130,
    "StocF-1465.mtx": 200,
    "circuit5M.mtx": 100,
    "dgreen.mtx": 300
}

PARTITION_DTYPE = torch.uint8

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
def run_partitioner(
    design_root,
    design,
    device,
    tag,
    result_root,
    N_CMA_ITE,
    KWAY,
    UB,
    seq_topm=128,
    seq_passes=6,
):

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
        "N_batch": batch_size_dict[design],
        "K": KWAY,
        "e": UB
    }

    hgr_file = config["hgr_file"]
    use_weight = config["use_weight"]
    N_batch = config["N_batch"]
    K = config["K"]
    e = config["e"]

    t_io0 = time.perf_counter()
    g = read_hgr(hgr_file)
    print(f"HGR IO (fast): {time.perf_counter() - t_io0:.3f}s")
    cell_cnt, net_cnt = g.num_nodes("cell"), g.num_nodes("net")
    print(g)
    g = g.to(device)

    W = cell_cnt
    th_l = W*(1/K - e)
    th_u = W*(1/K + e)
    print("Partition size constraints: ", th_l, th_u)

    # Bit-packed CUDA evaluator for unweighted 2-way (L_HG / Titan23 default path).
    # Falls back to DGL float32 message passing for weighted / K-way cases.
    cut_evaluator = None
    if K == 2 and not use_weight:
        t_ev0 = time.perf_counter()
        cut_evaluator = CutEvaluator(g, device, th_l, th_u)
        print(f"CutEvaluator init (CSR+JIT): {time.perf_counter() - t_ev0:.3f}s")

    # batch evaluation function
    def evaluation(g, population):
        cell_cnt, B = population.shape
        with torch.no_grad():
            if cut_evaluator is not None:
                return cut_evaluator(population)

            # Use float32 0/1 assignments for efficient DGL message passing
            assign_b = population.to(device=device, dtype=torch.float32)
            g.nodes["cell"].data["assign_b"] = assign_b

            # Aggregate number of part-1 pins per net per head
            g['connect'].update_all(fn.copy_u('assign_b', 'm'), fn.sum('m', 'net_sum1'), etype='connect')
            net_sum1 = g.nodes["net"].data["net_sum1"].to(torch.float32)

            # Cache per-net degree as [net_cnt, 1] and rely on broadcasting.
            # This avoids materializing a large [net_cnt, B] tensor on L_HG cases.
            deg_col = g.nodes["net"].data.get("deg_col", None)
            if deg_col is None:
                deg = g.in_degrees(torch.arange(g.num_nodes('net'), device=device), etype='connect').to(torch.float32)
                deg_col = deg.view(-1, 1)
                g.nodes["net"].data["deg_col"] = deg_col

            # A net is cut if it has at least one pin in both parts
            net_cut = (net_sum1 > 0.0) & (net_sum1 < deg_col)
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
        for _ in range(N_CMA_ITE):
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
                            if K == 2:
                                # Binary case: argmax([x^T w0, x^T w1]) is
                                # equivalent to x^T(w1-w0) > 0.  This avoids
                                # materializing logits with shape [N, B, 2].
                                w_delta = (assign_W_torch_s[..., 1] - assign_W_torch_s[..., 0]).sum(dim=1)
                                logits_delta = embed @ w_delta.T
                                solution = (logits_delta > 0).to(PARTITION_DTYPE)
                                del w_delta, logits_delta
                            else:
                                logits = torch.einsum('ne,paek->pank', embed, assign_W_torch_s)
                                logits = logits.sum(dim=1)
                                logits = logits.permute(1, 0, 2)
                                solution = torch.argmax(logits, 2).to(PARTITION_DTYPE)
                                del logits
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

                            del solution, score, assign_W_torch_s

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

    def compute_gains_batched(assign_b):
        """FM gains for [cell_cnt, B] assignments (used by sequence refine)."""
        cell_cnt_local = assign_b.shape[0]
        B_local = assign_b.shape[1]
        device_local = assign_b.device
        if cut_evaluator is not None:
            return cut_evaluator.compute_gains(assign_b)

        onehot = torch.zeros(cell_cnt_local, B_local, 2, device=device_local, dtype=torch.float32)
        onehot.scatter_(2, assign_b.long().unsqueeze(2), 1.0)
        g.nodes['cell'].data['part_onehot_b'] = onehot
        g['connect'].update_all(
            fn.copy_u('part_onehot_b', 'm'),
            fn.sum('m', 'net_counts_b'),
            etype='connect',
        )
        net_counts = g.nodes['net'].data['net_counts_b']
        net_cut = (net_counts[:, :, 0] > 0) & (net_counts[:, :, 1] > 0)
        pos0 = (net_cut & (net_counts[:, :, 0] == 1)).to(torch.float32)
        neg0 = ((~net_cut) & (net_counts[:, :, 1] == 0)).to(torch.float32)
        pos1 = (net_cut & (net_counts[:, :, 1] == 1)).to(torch.float32)
        neg1 = ((~net_cut) & (net_counts[:, :, 0] == 0)).to(torch.float32)
        cells_idx, nets_idx = g.in_edges(
            torch.arange(g.num_nodes('net'), device=device_local), etype='connect'
        )
        gain0 = torch.zeros(cell_cnt_local, B_local, device=device_local, dtype=torch.float32)
        gain1 = torch.zeros(cell_cnt_local, B_local, device=device_local, dtype=torch.float32)
        gain0.index_add_(0, cells_idx, pos0[nets_idx] - neg0[nets_idx])
        gain1.index_add_(0, cells_idx, pos1[nets_idx] - neg1[nets_idx])
        return torch.where(assign_b == 0, gain0, gain1)

    def cut_critical_mask_batched(assign_b):
        if cut_evaluator is not None:
            return cut_evaluator.cut_critical_mask(assign_b)
        return torch.ones_like(assign_b, dtype=torch.bool)

    def criticality_score_batched(assign_b):
        if cut_evaluator is not None:
            return cut_evaluator.criticality_score(assign_b)
        return torch.zeros(
            assign_b.shape[0], assign_b.shape[1], device=assign_b.device, dtype=torch.float32
        )

    def sequence_refine_batch(
        g,
        assign_batch,
        th_l,
        th_u,
        max_passes=6,
        top_m=128,
        refresh_every=8,
    ):
        """
        HyperG-inspired sequence multi-move refinement (independent reimpl.):
          1) Prefer cut-critical cells (iHyperG) among positive-gain moves.
          2) Build a gain-descending move sequence (tie-break by criticality).
          3) Walk the sequence with periodically refreshed gains under balance.
          4) Apply the best balanced prefix; accept only if true cut improves.
        """
        assign_batch = assign_batch.to(dtype=PARTITION_DTYPE)
        device_local = assign_batch.device
        cell_cnt = g.num_nodes('cell')
        B = assign_batch.shape[1]
        top_m = int(min(top_m, cell_cnt))

        with torch.no_grad():
            baseline_cut = evaluation(g, assign_batch).to(device_local).float()
        part0_sizes = (assign_batch == 0).sum(dim=0).float()
        improved_any = torch.zeros(B, dtype=torch.bool, device=device_local)
        cols = torch.arange(B, device=device_local)

        if cut_evaluator is not None and getattr(cut_evaluator, "_skewed", False):
            top_m = min(top_m, 64)
            refresh_every = max(refresh_every, 16)

        for _pass in range(max_passes):
            gains = compute_gains_batched(assign_batch)
            crit = cut_critical_mask_batched(assign_batch)
            crit_score = criticality_score_batched(assign_batch)

            # Prefer cut-critical / high criticality, but keep all +gain cells.
            rank = gains + 1e3 * crit.float() + 1e-3 * crit_score
            rank = rank.masked_fill(gains <= 0, float('-inf'))

            top_vals, top_idx = torch.topk(rank, k=top_m, dim=0)
            working = assign_batch.clone()
            w_p0 = part0_sizes.clone()
            accepted_rows = [[] for _ in range(B)]
            accepted_gains = [[] for _ in range(B)]
            live_gains = gains.clone()

            for step in range(top_m):
                if step > 0 and (step % refresh_every == 0):
                    live_gains = compute_gains_batched(working)

                rows = top_idx[step]
                step_gain = live_gains[rows, cols]
                cur_part = working[rows, cols]
                new_p0 = torch.where(cur_part == 0, w_p0 - 1.0, w_p0 + 1.0)
                new_p1 = float(cell_cnt) - new_p0
                bal_ok = (new_p0 >= th_l) & (new_p0 <= th_u) & (new_p1 >= th_l) & (new_p1 <= th_u)
                take = (step_gain > 0) & bal_ok & (top_vals[step] > float('-inf'))
                if not take.any():
                    continue

                t_rows = rows[take]
                t_cols = cols[take]
                working[t_rows, t_cols] = 1 - cur_part[take]
                delta = torch.where(cur_part[take] == 0, -1.0, 1.0)
                w_p0[take] = w_p0[take] + delta

                for b in take.nonzero(as_tuple=False).flatten().tolist():
                    accepted_rows[b].append(int(rows[b].item()))
                    accepted_gains[b].append(float(step_gain[b].item()))

            tentative = assign_batch.clone()
            any_move = False
            for b in range(B):
                rows_b = accepted_rows[b]
                gains_b = accepted_gains[b]
                if not rows_b:
                    continue
                cum = 0.0
                best_j = -1
                best_cum = 0.0
                for j, gv in enumerate(gains_b):
                    cum += gv
                    if cum > best_cum:
                        best_cum = cum
                        best_j = j
                if best_j < 0:
                    continue
                for j in range(best_j + 1):
                    tentative[rows_b[j], b] = 1 - assign_batch[rows_b[j], b]
                any_move = True

            if not any_move:
                break

            with torch.no_grad():
                new_scores = evaluation(g, tentative).to(device_local).float()
            accept = new_scores < baseline_cut
            if not accept.any():
                break

            assign_batch[:, accept] = tentative[:, accept]
            baseline_cut[accept] = new_scores[accept]
            part0_sizes[accept] = (assign_batch[:, accept] == 0).sum(dim=0).float()
            improved_any[accept] = True

        return assign_batch, improved_any

    # construct embedding spaces
    embed_list = []
    k = 48
    eigvals, eigvecs_cells_torch = bipartite_spectral_embeddings_from_dgl(
        g, etype=('cell', 'connect', 'net'), k=k, which='SA', device=device
    )
    print("Bipartite graph eigvals:", eigvals)
    mask = eigvals > 1e-6
    eigvals = eigvals[mask]
    eigvecs_cells_torch = eigvecs_cells_torch[:, mask]
    embed_list += [
        eigvecs_cells_torch[:, :4],
        eigvecs_cells_torch[:, :8],
        eigvecs_cells_torch[:, :16],
        eigvecs_cells_torch[:, :24],
        eigvecs_cells_torch[:, :36],
    ]

    best_solution, best_score = CMA_ES_iterations(g, embed_list)
    print("Best score after CMA-ES on all embedding spaces: ", best_score.min())

    scores_sorted, indices = torch.sort(best_score)
    best_solution = best_solution[:, indices]
    best_score = scores_sorted

    total_edge_degree = int(g.num_edges(etype='connect'))
    topK = 32
    if total_edge_degree >= 100_000_000:
        topK = 2
    elif total_edge_degree >= 50_000_000:
        topK = 8

    print(
        f"Sequence refine topK={topK} seq_topm={seq_topm} seq_passes={seq_passes} "
        f"(total_edge_degree={total_edge_degree})"
    )
    best_assign = best_solution[:, :topK].to(device)
    pre_cut = evaluation(g, best_assign)
    print(f"Pre-refine best cut: {float(pre_cut.min()):.1f}")
    try:
        t0 = time.perf_counter()
        refined_assign, did_improve = sequence_refine_batch(
            g, best_assign, th_l, th_u, max_passes=seq_passes, top_m=seq_topm
        )
        print(
            f"Sequence refine: improved_heads={int(did_improve.sum())} "
            f"time={time.perf_counter() - t0:.3f}s"
        )
    except torch.cuda.OutOfMemoryError as oom:
        print(f"Refinement skipped due to CUDA OOM: {oom}")
        torch.cuda.empty_cache()
        refined_assign = best_assign
        did_improve = torch.zeros(topK, dtype=torch.bool, device=device)

    best_solution[:, :topK] = refined_assign.to(best_solution.device)
    best_score[:topK] = evaluation(g, best_solution[:, :topK])
    print(
        f"Post-refine best cut: {float(best_score.min()):.1f} "
        f"(improved_heads={int(did_improve.sum())})"
    )
    scores_sorted, indices = torch.sort(best_score)
    best_solution = best_solution[:, indices]
    best_score = scores_sorted

    torch.save(best_solution[:, 0].to("cpu"), config["solution_file"])
    torch.save(best_score.min().detach().cpu(), config["score_file"])
    return best_score.min().item()


def main():
    parser = argparse.ArgumentParser(description="HySpecPro (sequence local search)")
    parser.add_argument("--design_root", type=str, default="./benchmark/Titan23_benchmark/", help="Root path to the hgr files")
    parser.add_argument("--design", type=str, default="sparcT1_core", help="Design name")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.add_argument("--tag", type=str, default="v1", help="Tag to differentiate different runs")
    parser.add_argument("--result_root", type=str, default="./results/", help="Root path to the results")
    parser.add_argument("--N_CMA_ITE", type=int, default=5, help="Number of CMA-ES iterations")
    parser.add_argument("--KWAY", type=int, default=2, help="Number of partitions")
    parser.add_argument("--UB", type=float, default=0.02, help="Upper bound for the partition size")
    parser.add_argument("--seq_topm", type=int, default=128, help="Max move-sequence length per pass")
    parser.add_argument("--seq_passes", type=int, default=6, help="Sequence refine passes")
    args = parser.parse_args()

    best_score = run_partitioner(
        args.design_root,
        args.design,
        args.device,
        args.tag,
        args.result_root,
        args.N_CMA_ITE,
        args.KWAY,
        args.UB,
        seq_topm=args.seq_topm,
        seq_passes=args.seq_passes,
    )
    print(f"Final best score: {best_score}")


if __name__ == "__main__":
    main()
