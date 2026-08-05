# Appendix C. Environment and reproduction

All sessions ran on RunPod instances with NVIDIA A40 48GB GPUs and an Intel Xeon Gold
6342 (96 logical CPUs, two NUMA nodes). The driver was standardised to the 580 series;
the exact build differed between instances but the CUDA runtime did not.

## Hardware

| Session | GPUs | Region | Driver / CUDA |
|---|---|---|---|
| A | 1 x A40 48GB | eu-se-1 | 580.95.05 / CUDA 13.0 |
| B | 1 x A40 48GB | ca-mtl-1 | 580.126.09 / CUDA 13.0 |
| C | 4 x A40 48GB | not recorded | 580.159.04 / CUDA 13.0 |
| D | 4 x A40 48GB | not recorded | 580.159.04 / CUDA 13.0 |

The region of Sessions C and D was not captured at run time and is not recoverable from
the archived logs. This does not affect any comparison in the report: the GPU-count
series (G1/G2/G4) is measured entirely within Session C (§3.3), and the one comparison
that does cross instances — P1 against the Session C series — is bounded by an explicit
tolerance rather than by an assumption of identical hosts (§3.5).

## GPU topology

Recorded with `nvidia-smi topo -m` on the Session C host. Session D reports the same
matrix except that GPU2–GPU3 are PXB rather than PIX; the GPU0–GPU1 pair used for
tp = 2 and pp = 2 is PXB in both sessions.

```
        GPU0    GPU1    GPU2    GPU3    NIC0    NIC1    CPU Affinity    NUMA Affinity
GPU0     X      PXB     SYS     SYS     SYS     SYS     0-23,48-71      0
GPU1    PXB      X      SYS     SYS     SYS     SYS     0-23,48-71      0
GPU2    SYS     SYS      X      PIX     NODE    NODE    24-47,72-95     1
GPU3    SYS     SYS     PIX      X      NODE    NODE    24-47,72-95     1
NIC0    SYS     SYS     NODE    NODE     X      PIX
NIC1    SYS     SYS     NODE    NODE    PIX      X

PIX  = at most a single PCIe bridge
PXB  = multiple PCIe bridges, without traversing the PCIe host bridge
NODE = PCIe plus an interconnect between host bridges within one NUMA node
SYS  = PCIe plus the SMP interconnect between NUMA nodes (QPI/UPI)

NIC0: rocep177s0f0    NIC1: rocep177s0f1
```

This is the evidence for §5.4. GPU0 and GPU1 sit on NUMA node 0 and communicate over
PXB, so the all-reduce at tp = 2 stays within one node. GPU2 and GPU3 sit on node 1, so
any aggregation at tp = 4 necessarily crosses SYS. There are no NVLink (`NV#`) entries
in the matrix, which is why the peer-to-peer workaround below was required.

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
