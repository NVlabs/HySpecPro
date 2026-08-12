// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Batched 2-way hypergraph cut evaluation with bit-packed assignments.
//
// Graph CSR-by-net: pins of net e are cell_idx[net_ptr[e] : net_ptr[e+1]].
// Assignments are packed as uint32 bit-words, layout [cell_cnt, nwords]
// (row-major), where bit b of the population is stored in word (b>>5) at
// bit-position (b & 31).
//
// Cut test (per head bit):
//   mixed  => OR==1, AND==0  => cut bit = OR & ~AND
//
// Memory hierarchy:
// - Pack uint8->[words] once per eval with coalesced row writes.
// - One thread per net (grid-stride) streams pins once; OR/AND accumulators
//   stay in registers (nwords <= 16 => B<=512).
// - Traffic ~B/8 bytes/pin vs 4B for float32 DGL message passing.
// - Block shared histogram + one atomic flush per head.

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <algorithm>

namespace {

constexpr int MAX_WORDS = 32;  // B <= 1024
constexpr int BLOCK_THREADS = 256;

__global__ void pack_u8_to_words_kernel(
    const uint8_t* __restrict__ assign,  // [N, B]
    uint32_t* __restrict__ words,        // [N, nwords]
    int N, int B, int nwords)
{
    for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < N;
         c += blockDim.x * gridDim.x) {
        const uint8_t* row = assign + static_cast<int64_t>(c) * B;
        uint32_t* out = words + static_cast<int64_t>(c) * nwords;
        for (int w = 0; w < nwords; ++w) out[w] = 0u;
        for (int b = 0; b < B; ++b) {
            if (row[b]) {
                out[b >> 5] |= (1u << (b & 31));
            }
        }
    }
}

// Low-degree nets: one thread per net (via compacted low_net_ids).
__global__ void cut_count_lowdeg_bitpack_kernel(
    const int32_t* __restrict__ low_net_ids,  // [n_low]
    int n_low,
    const int32_t* __restrict__ net_ptr,
    const int32_t* __restrict__ cell_idx,
    const uint32_t* __restrict__ assign_words,  // [cell_cnt, nwords]
    const int32_t* __restrict__ deg,
    int32_t* __restrict__ N_cut,                // [B]
    int nwords,
    int B,
    uint32_t last_mask)
{
    extern __shared__ int32_t sh_hist[];  // [B]

    for (int i = threadIdx.x; i < B; i += blockDim.x) sh_hist[i] = 0;
    __syncthreads();

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_low;
         i += blockDim.x * gridDim.x) {
        const int e = low_net_ids[i];
        const int start = net_ptr[e];
        const int end = net_ptr[e + 1];
        const int d = deg[e];
        if (d <= 1) continue;

        uint32_t or_acc[MAX_WORDS];
        uint32_t and_acc[MAX_WORDS];
        #pragma unroll
        for (int w = 0; w < MAX_WORDS; ++w) {
            or_acc[w] = 0u;
            and_acc[w] = 0xFFFFFFFFu;
        }
        if (nwords > 0) and_acc[nwords - 1] = last_mask;

        for (int p = start; p < end; ++p) {
            const uint32_t* row =
                assign_words + static_cast<int64_t>(cell_idx[p]) * nwords;
            #pragma unroll
            for (int w = 0; w < MAX_WORDS; ++w) {
                if (w < nwords) {
                    const uint32_t v = row[w];
                    or_acc[w] |= v;
                    and_acc[w] &= v;
                }
            }
        }

        #pragma unroll
        for (int w = 0; w < MAX_WORDS; ++w) {
            if (w >= nwords) break;
            uint32_t cutbits = or_acc[w] & ~and_acc[w];
            if (w == nwords - 1) cutbits &= last_mask;
            while (cutbits) {
                const int bit = __ffs(cutbits) - 1;
                cutbits &= cutbits - 1u;
                const int b = (w << 5) + bit;
                if (b < B) atomicAdd(&sh_hist[b], 1);
            }
        }
    }
    __syncthreads();

    for (int i = threadIdx.x; i < B; i += blockDim.x) {
        const int v = sh_hist[i];
        if (v) atomicAdd(&N_cut[i], v);
    }
}

// High-degree nets: one block per net, cooperative pin OR/AND reduction.
// Critical for skewed graphs (e.g. circuit5M has nets with deg ~1.3M).
__global__ void cut_count_highdeg_bitpack_kernel(
    const int32_t* __restrict__ high_net_ids,  // [n_high]
    int n_high,
    const int32_t* __restrict__ net_ptr,
    const int32_t* __restrict__ cell_idx,
    const uint32_t* __restrict__ assign_words,
    const int32_t* __restrict__ deg,
    int32_t* __restrict__ N_cut,
    int nwords,
    int B,
    uint32_t last_mask)
{
    const int h = blockIdx.x;
    if (h >= n_high) return;
    const int e = high_net_ids[h];
    const int d = deg[e];
    if (d <= 1) return;

    const int start = net_ptr[e];
    const int end = net_ptr[e + 1];

    // Dynamic: [nwords or][nwords and][B hist]
    extern __shared__ char smem[];
    uint32_t* sh_or = reinterpret_cast<uint32_t*>(smem);
    uint32_t* sh_and = sh_or + nwords;
    int32_t* sh_hist = reinterpret_cast<int32_t*>(sh_and + nwords);

    for (int w = threadIdx.x; w < nwords; w += blockDim.x) {
        sh_or[w] = 0u;
        sh_and[w] = (w == nwords - 1) ? last_mask : 0xFFFFFFFFu;
    }
    for (int i = threadIdx.x; i < B; i += blockDim.x) sh_hist[i] = 0;
    __syncthreads();

    uint32_t lor[MAX_WORDS];
    uint32_t land[MAX_WORDS];
    #pragma unroll
    for (int w = 0; w < MAX_WORDS; ++w) {
        lor[w] = 0u;
        land[w] = 0xFFFFFFFFu;
    }
    if (nwords > 0) land[nwords - 1] = last_mask;

    for (int p = start + threadIdx.x; p < end; p += blockDim.x) {
        const uint32_t* row =
            assign_words + static_cast<int64_t>(cell_idx[p]) * nwords;
        #pragma unroll
        for (int w = 0; w < MAX_WORDS; ++w) {
            if (w < nwords) {
                const uint32_t v = row[w];
                lor[w] |= v;
                land[w] &= v;
            }
        }
    }

    #pragma unroll
    for (int w = 0; w < MAX_WORDS; ++w) {
        if (w < nwords) {
            atomicOr(&sh_or[w], lor[w]);
            atomicAnd(&sh_and[w], land[w]);
        }
    }
    __syncthreads();

    for (int w = threadIdx.x; w < nwords; w += blockDim.x) {
        uint32_t cutbits = sh_or[w] & ~sh_and[w];
        if (w == nwords - 1) cutbits &= last_mask;
        while (cutbits) {
            const int bit = __ffs(cutbits) - 1;
            cutbits &= cutbits - 1u;
            const int b = (w << 5) + bit;
            if (b < B) atomicAdd(&sh_hist[b], 1);
        }
    }
    __syncthreads();

    for (int i = threadIdx.x; i < B; i += blockDim.x) {
        const int v = sh_hist[i];
        if (v) atomicAdd(&N_cut[i], v);
    }
}

__global__ void part_sum_bitpack_kernel(
    const uint32_t* __restrict__ assign_words,
    int32_t* __restrict__ ones,
    int cell_cnt,
    int nwords,
    int B,
    uint32_t last_mask)
{
    extern __shared__ int32_t sh_hist[];
    for (int i = threadIdx.x; i < B; i += blockDim.x) sh_hist[i] = 0;
    __syncthreads();

    for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < cell_cnt;
         c += blockDim.x * gridDim.x) {
        const uint32_t* row =
            assign_words + static_cast<int64_t>(c) * nwords;
        #pragma unroll
        for (int w = 0; w < MAX_WORDS; ++w) {
            if (w >= nwords) break;
            uint32_t v = row[w];
            if (w == nwords - 1) v &= last_mask;
            while (v) {
                const int bit = __ffs(v) - 1;
                v &= v - 1u;
                const int b = (w << 5) + bit;
                if (b < B) atomicAdd(&sh_hist[b], 1);
            }
        }
    }
    __syncthreads();

    for (int i = threadIdx.x; i < B; i += blockDim.x) {
        const int v = sh_hist[i];
        if (v) atomicAdd(&ones[i], v);
    }
}

// ---------------------------------------------------------------------------
// FM/KL batched gain computation
// ---------------------------------------------------------------------------
// FM_MAX_B bounds register tiles. FM typically uses topK<=32 heads.
constexpr int FM_MAX_B = 64;

// Pass 1a: low-degree nets — one thread per net (via compacted ids).
__global__ void fm_net_contrib_lowdeg_kernel(
    const int32_t* __restrict__ low_net_ids,
    int n_low,
    const int32_t* __restrict__ net_ptr,
    const int32_t* __restrict__ cell_idx,
    const uint8_t* __restrict__ assign,  // [cell_cnt, B]
    const int32_t* __restrict__ deg,
    int8_t* __restrict__ contrib0,       // [net_cnt, B]
    int8_t* __restrict__ contrib1,       // [net_cnt, B]
    int B)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_low;
         i += blockDim.x * gridDim.x) {
        const int e = low_net_ids[i];
        const int start = net_ptr[e];
        const int end = net_ptr[e + 1];
        const int d = deg[e];

        int sum1[FM_MAX_B];
        #pragma unroll
        for (int b = 0; b < FM_MAX_B; ++b) sum1[b] = 0;

        for (int p = start; p < end; ++p) {
            const uint8_t* row =
                assign + static_cast<int64_t>(cell_idx[p]) * B;
            #pragma unroll
            for (int b = 0; b < FM_MAX_B; ++b) {
                if (b < B) sum1[b] += static_cast<int>(row[b]);
            }
        }

        int8_t* c0 = contrib0 + static_cast<int64_t>(e) * B;
        int8_t* c1 = contrib1 + static_cast<int64_t>(e) * B;
        #pragma unroll
        for (int b = 0; b < FM_MAX_B; ++b) {
            if (b >= B) break;
            const int s = sum1[b];
            const bool cut = (s > 0) && (s < d);
            const int pos0 = (cut && (s == d - 1)) ? 1 : 0;
            const int neg0 = ((!cut) && (s == 0)) ? 1 : 0;
            const int pos1 = (cut && (s == 1)) ? 1 : 0;
            const int neg1 = ((!cut) && (s == d)) ? 1 : 0;
            c0[b] = static_cast<int8_t>(pos0 - neg0);
            c1[b] = static_cast<int8_t>(pos1 - neg1);
        }
    }
}

// Pass 1b: high-degree nets — one block per net, cooperative sum reduction.
__global__ void fm_net_contrib_highdeg_kernel(
    const int32_t* __restrict__ high_net_ids,
    int n_high,
    const int32_t* __restrict__ net_ptr,
    const int32_t* __restrict__ cell_idx,
    const uint8_t* __restrict__ assign,
    const int32_t* __restrict__ deg,
    int8_t* __restrict__ contrib0,
    int8_t* __restrict__ contrib1,
    int B)
{
    const int h = blockIdx.x;
    if (h >= n_high) return;
    const int e = high_net_ids[h];
    const int d = deg[e];
    const int start = net_ptr[e];
    const int end = net_ptr[e + 1];

    extern __shared__ int sh_sum[];  // [B]
    for (int b = threadIdx.x; b < B; b += blockDim.x) sh_sum[b] = 0;
    __syncthreads();

    int local[FM_MAX_B];
    #pragma unroll
    for (int b = 0; b < FM_MAX_B; ++b) local[b] = 0;

    for (int p = start + threadIdx.x; p < end; p += blockDim.x) {
        const uint8_t* row =
            assign + static_cast<int64_t>(cell_idx[p]) * B;
        #pragma unroll
        for (int b = 0; b < FM_MAX_B; ++b) {
            if (b < B) local[b] += static_cast<int>(row[b]);
        }
    }

    #pragma unroll
    for (int b = 0; b < FM_MAX_B; ++b) {
        if (b < B && local[b]) atomicAdd(&sh_sum[b], local[b]);
    }
    __syncthreads();

    for (int b = threadIdx.x; b < B; b += blockDim.x) {
        const int s = sh_sum[b];
        const bool cut = (s > 0) && (s < d);
        const int pos0 = (cut && (s == d - 1)) ? 1 : 0;
        const int neg0 = ((!cut) && (s == 0)) ? 1 : 0;
        const int pos1 = (cut && (s == 1)) ? 1 : 0;
        const int neg1 = ((!cut) && (s == d)) ? 1 : 0;
        contrib0[static_cast<int64_t>(e) * B + b] =
            static_cast<int8_t>(pos0 - neg0);
        contrib1[static_cast<int64_t>(e) * B + b] =
            static_cast<int8_t>(pos1 - neg1);
    }
}

// Pass 2: for each cell, sum contribs over incident nets (atomic-free).
// gains[c,b] = sum_e (assign[c,b] ? contrib1[e,b] : contrib0[e,b])
__global__ void fm_scatter_gains_kernel(
    const int32_t* __restrict__ cell_ptr,
    const int32_t* __restrict__ net_idx,
    const uint8_t* __restrict__ assign,   // [cell_cnt, B]
    const int8_t* __restrict__ contrib0,  // [net_cnt, B]
    const int8_t* __restrict__ contrib1,  // [net_cnt, B]
    float* __restrict__ gains,            // [cell_cnt, B]
    int cell_cnt,
    int B)
{
    for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < cell_cnt;
         c += blockDim.x * gridDim.x) {
        const int start = cell_ptr[c];
        const int end = cell_ptr[c + 1];

        int acc[FM_MAX_B];
        #pragma unroll
        for (int b = 0; b < FM_MAX_B; ++b) acc[b] = 0;

        const uint8_t* arow = assign + static_cast<int64_t>(c) * B;
        for (int p = start; p < end; ++p) {
            const int e = net_idx[p];
            const int8_t* c0 = contrib0 + static_cast<int64_t>(e) * B;
            const int8_t* c1 = contrib1 + static_cast<int64_t>(e) * B;
            #pragma unroll
            for (int b = 0; b < FM_MAX_B; ++b) {
                if (b < B) {
                    acc[b] += arow[b] ? static_cast<int>(c1[b])
                                      : static_cast<int>(c0[b]);
                }
            }
        }

        float* gout = gains + static_cast<int64_t>(c) * B;
        #pragma unroll
        for (int b = 0; b < FM_MAX_B; ++b) {
            if (b < B) gout[b] = static_cast<float>(acc[b]);
        }
    }
}

}  // namespace

std::vector<torch::Tensor> evaluate_cut_cuda(
    torch::Tensor net_ptr,
    torch::Tensor cell_idx,
    torch::Tensor assign_u8,
    torch::Tensor deg,
    torch::Tensor low_net_ids,
    torch::Tensor high_net_ids,
    double th_l,
    double th_u)
{
    TORCH_CHECK(net_ptr.is_cuda() && cell_idx.is_cuda() && assign_u8.is_cuda() &&
                    deg.is_cuda() && low_net_ids.is_cuda() && high_net_ids.is_cuda(),
                "all tensors must be CUDA");
    TORCH_CHECK(assign_u8.dtype() == torch::kUInt8, "assign must be uint8");
    TORCH_CHECK(net_ptr.dtype() == torch::kInt32 && cell_idx.dtype() == torch::kInt32 &&
                deg.dtype() == torch::kInt32 && low_net_ids.dtype() == torch::kInt32 &&
                high_net_ids.dtype() == torch::kInt32);
    TORCH_CHECK(assign_u8.dim() == 2 && assign_u8.is_contiguous());
    TORCH_CHECK(net_ptr.is_contiguous() && cell_idx.is_contiguous() &&
                deg.is_contiguous() && low_net_ids.is_contiguous() &&
                high_net_ids.is_contiguous());

    const int cell_cnt = static_cast<int>(assign_u8.size(0));
    const int B = static_cast<int>(assign_u8.size(1));
    const int net_cnt = static_cast<int>(deg.size(0));
    const int n_low = static_cast<int>(low_net_ids.size(0));
    const int n_high = static_cast<int>(high_net_ids.size(0));
    TORCH_CHECK(B <= MAX_WORDS * 32, "B too large for bitpack kernel");
    TORCH_CHECK(net_ptr.size(0) == net_cnt + 1);

    const int nwords = (B + 31) / 32;
    const uint32_t last_mask =
        (B % 32 == 0) ? 0xFFFFFFFFu : ((1u << (B % 32)) - 1u);

    auto opts_i =
        torch::TensorOptions().dtype(torch::kInt32).device(assign_u8.device());
    torch::Tensor words = torch::empty({cell_cnt, nwords}, opts_i);
    torch::Tensor N_cut = torch::zeros({B}, opts_i);

    const int grids_cell =
        std::min((cell_cnt + BLOCK_THREADS - 1) / BLOCK_THREADS, 4096);
    const int grids_low =
        std::min((n_low + BLOCK_THREADS - 1) / BLOCK_THREADS, 4096);
    const int smem_hist = B * static_cast<int>(sizeof(int32_t));
    const int smem_high =
        nwords * static_cast<int>(sizeof(uint32_t)) * 2 + smem_hist;

    if (cell_cnt > 0 && B > 0) {
        pack_u8_to_words_kernel<<<grids_cell, BLOCK_THREADS>>>(
            assign_u8.data_ptr<uint8_t>(),
            reinterpret_cast<uint32_t*>(words.data_ptr<int32_t>()),
            cell_cnt, B, nwords);
    }

    if (n_low > 0 && B > 0) {
        cut_count_lowdeg_bitpack_kernel<<<grids_low, BLOCK_THREADS, smem_hist>>>(
            low_net_ids.data_ptr<int32_t>(), n_low,
            net_ptr.data_ptr<int32_t>(), cell_idx.data_ptr<int32_t>(),
            reinterpret_cast<const uint32_t*>(words.data_ptr<int32_t>()),
            deg.data_ptr<int32_t>(), N_cut.data_ptr<int32_t>(), nwords, B,
            last_mask);
    }

    if (n_high > 0 && B > 0) {
        cut_count_highdeg_bitpack_kernel<<<n_high, BLOCK_THREADS, smem_high>>>(
            high_net_ids.data_ptr<int32_t>(), n_high,
            net_ptr.data_ptr<int32_t>(), cell_idx.data_ptr<int32_t>(),
            reinterpret_cast<const uint32_t*>(words.data_ptr<int32_t>()),
            deg.data_ptr<int32_t>(), N_cut.data_ptr<int32_t>(), nwords, B,
            last_mask);
    }

    // Balance: direct uint8 reduction is far cheaper than bit-unpack popcounts
    // on large cell counts (e.g. circuit5M has 5.5M cells).
    auto ones_f = assign_u8.to(torch::kFloat32).sum(/*dim=*/0);
    auto N_cut_f = N_cut.to(torch::kFloat32);
    auto zeros_f = static_cast<float>(cell_cnt) - ones_f;
    auto weight_mat = torch::stack({zeros_f, ones_f}, /*dim=*/1);
    auto penalty = 50.0f * torch::sum(
        torch::relu(weight_mat - static_cast<float>(th_u)) +
            torch::relu(static_cast<float>(th_l) - weight_mat),
        /*dim=*/1);
    auto score = N_cut_f + penalty;
    return {score, N_cut_f, ones_f};
}

torch::Tensor compute_fm_gains_cuda(
    torch::Tensor net_ptr,
    torch::Tensor cell_idx,
    torch::Tensor cell_ptr,
    torch::Tensor net_idx,
    torch::Tensor assign_u8,
    torch::Tensor deg,
    torch::Tensor low_net_ids,
    torch::Tensor high_net_ids)
{
    TORCH_CHECK(net_ptr.is_cuda() && cell_idx.is_cuda() && cell_ptr.is_cuda() &&
                net_idx.is_cuda() && assign_u8.is_cuda() && deg.is_cuda() &&
                low_net_ids.is_cuda() && high_net_ids.is_cuda());
    TORCH_CHECK(assign_u8.dtype() == torch::kUInt8);
    TORCH_CHECK(assign_u8.dim() == 2 && assign_u8.is_contiguous());
    TORCH_CHECK(net_ptr.dtype() == torch::kInt32 && cell_idx.dtype() == torch::kInt32 &&
                cell_ptr.dtype() == torch::kInt32 && net_idx.dtype() == torch::kInt32 &&
                deg.dtype() == torch::kInt32 && low_net_ids.dtype() == torch::kInt32 &&
                high_net_ids.dtype() == torch::kInt32);

    const int cell_cnt = static_cast<int>(assign_u8.size(0));
    const int B = static_cast<int>(assign_u8.size(1));
    const int net_cnt = static_cast<int>(deg.size(0));
    const int n_low = static_cast<int>(low_net_ids.size(0));
    const int n_high = static_cast<int>(high_net_ids.size(0));
    TORCH_CHECK(B <= FM_MAX_B, "B too large for FM gains kernel; chunk in Python");
    TORCH_CHECK(net_ptr.size(0) == net_cnt + 1);
    TORCH_CHECK(cell_ptr.size(0) == cell_cnt + 1);

    auto opts_i8 =
        torch::TensorOptions().dtype(torch::kInt8).device(assign_u8.device());
    auto opts_f =
        torch::TensorOptions().dtype(torch::kFloat32).device(assign_u8.device());

    torch::Tensor contrib0 = torch::empty({net_cnt, B}, opts_i8);
    torch::Tensor contrib1 = torch::empty({net_cnt, B}, opts_i8);
    torch::Tensor gains = torch::empty({cell_cnt, B}, opts_f);

    const int grids_low =
        std::min((n_low + BLOCK_THREADS - 1) / BLOCK_THREADS, 4096);
    const int grids_cell =
        std::min((cell_cnt + BLOCK_THREADS - 1) / BLOCK_THREADS, 4096);
    const int smem_high = B * static_cast<int>(sizeof(int));

    if (n_low > 0 && B > 0) {
        fm_net_contrib_lowdeg_kernel<<<grids_low, BLOCK_THREADS>>>(
            low_net_ids.data_ptr<int32_t>(), n_low,
            net_ptr.data_ptr<int32_t>(), cell_idx.data_ptr<int32_t>(),
            assign_u8.data_ptr<uint8_t>(), deg.data_ptr<int32_t>(),
            contrib0.data_ptr<int8_t>(), contrib1.data_ptr<int8_t>(), B);
    }

    if (n_high > 0 && B > 0) {
        fm_net_contrib_highdeg_kernel<<<n_high, BLOCK_THREADS, smem_high>>>(
            high_net_ids.data_ptr<int32_t>(), n_high,
            net_ptr.data_ptr<int32_t>(), cell_idx.data_ptr<int32_t>(),
            assign_u8.data_ptr<uint8_t>(), deg.data_ptr<int32_t>(),
            contrib0.data_ptr<int8_t>(), contrib1.data_ptr<int8_t>(), B);
    }

    if (cell_cnt > 0 && B > 0) {
        fm_scatter_gains_kernel<<<grids_cell, BLOCK_THREADS>>>(
            cell_ptr.data_ptr<int32_t>(), net_idx.data_ptr<int32_t>(),
            assign_u8.data_ptr<uint8_t>(), contrib0.data_ptr<int8_t>(),
            contrib1.data_ptr<int8_t>(), gains.data_ptr<float>(), cell_cnt,
            B);
    }

    return gains;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("evaluate_cut_cuda", &evaluate_cut_cuda,
          "Batched binary cut + balance evaluation (bit-packed)");
    m.def("compute_fm_gains_cuda", &compute_fm_gains_cuda,
          "Batched FM/KL gain computation (uint8 CSR)");
}
