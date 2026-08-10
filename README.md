# HySpecPro


HySpecPro is a GPU-accelerated single-level hypergraph partitioner based on
spectral embedding and projection optimization.  This release focuses on
2-way partitioning with imbalance tolerance `UB`.

## Requirements

Use Python 3.10 with the following packages installed:

```text
torch
dgl
numpy
scipy
cupy
cma
```

## Benchmark Layout

Place benchmark `.hgr` files under:

```text
benchmarks/Titan23_benchmark/
benchmarks/L_HG_benchmark/
```

The batch scripts assume these paths by default, but they can be overridden with
`DESIGN_ROOT`.

## Run One Design

```bash
python HySpecPro.py \
  --design_root benchmarks/Titan23_benchmark/ \
  --result_root ./results/ \
  --design sparcT1_core \
  --device cuda:0 \
  --tag test_run \
  --N_CMA_ITE 5 \
  --KWAY 2 \
  --UB 0.02
```

The script writes:

```text
results/<tag>_<design>_best_solution.pt
results/<tag>_<design>_best_score.pt
```

## Batch Runs

Run all Titan23 designs:

```bash
./run_titan23.sh
```

Run the L_HG benchmark subset:

```bash
./run_LHG.sh
```

Both scripts write a CSV summary with cut size, runtime, status, and paths to
the saved solution/score files.

## Common Overrides

```bash
PY=/path/to/python \
DESIGN_ROOT=/path/to/benchmark/root/ \
RESULT_ROOT=./results/ \
TAG=my_run \
UB=0.02 \
KWAY=2 \
N_CMA_ITE=5 \
DEVICES_OVERRIDE="cuda:0 cuda:1 cuda:2 cuda:3" \
./run_titan23.sh
```
