# Tilelang-AttnRes

This repository provides a highly optimized, deeply fused GPU kernel implemented in [TileLang](https://github.com/tile-ai/tilelang) for attention residual architectures.

### 1. Speed test (on RTX 5080)

Tested with input shape `[1280, 8, 7168]` and `bfloat16` precision.

| Implementation | Latency (ms) | Speedup |
| :--- | :--- | :--- |
| **PyTorch Eager** | 4.708 | 1.00x (Baseline) |
| **Torch Compile** | 1.111 | 4.23x |
| **TileLang (Ours)** | **0.238** | **19.78x** |