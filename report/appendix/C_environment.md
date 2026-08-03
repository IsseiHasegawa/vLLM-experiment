# Appendix C. Environment and reproduction

<!-- TODO: fill from results/raw/session*/env_freeze.txt, lscpu, nvidia_topo.txt. -->

## Hardware

| Session | GPUs | Region | Driver / CUDA |
|---|---|---|---|
| A | 1 x A40 48GB | eu-se-1 | 580 series / CUDA 13.0 |
| B | 1 x A40 48GB | ca-mtl-1 | 580 series / CUDA 13.0 |
| C | 4 x A40 48GB | <!-- TODO --> | 580.159.04 / CUDA 13.0 |

## GPU topology

<!-- TODO: paste results/raw/sessionC/nvidia_topo.txt. Required to support 5.4:
     GPU0-1 are PXB (same NUMA node), GPU0-3 are SYS (crossing NUMA). -->

## Software

- torch 2.11.0+cu130
- vLLM `d4e0675a7` (fork of `702f4814f`)
- Installed with `VLLM_USE_PRECOMPILED=1` and `--no-build-isolation`, so all sessions
  share identical CUDA kernels.

## Multi-GPU workaround

For tp>1 only, `NCCL_P2P_DISABLE=1` and `--disable-custom-all-reduce` were required; the
communicator otherwise hangs during setup on this host, which has no NVLink, a split NUMA
topology, and RoCE NICs. These settings were scoped to tp>1 server processes so that tp=1
commands remain byte-identical to Sessions A and B.

## Reproduction

```
git clone https://github.com/IsseiHasegawa/vLLM-experiment
python3 scripts/make_matrix.py            # regenerates configs/matrix.csv
python3 scripts/run_experiments.py --session A --only A1a,C1off,S1,S2,I1,I2,A1b,P0,S3
python3 scripts/plots/make_figures.py     # regenerates all figures from results/
```
