# HySpecPro
HySPecPro: HySpecPro is a single-level hypergraph partitioner that performs end-to-end optimization in a spectral embedding space. HySpecPro constructs embeddings from a bipartite Laplacian and performs efficient projection-based search, supported by a fully GPU-accelerated implementation.

1. Update the configs (e.g., UB, KWAY, design_root, and result_root) in HySpecPro.py
2. python HySpecPro.py --design <DESIGN NAME> --device cuda:0 --tag test --seed 0
