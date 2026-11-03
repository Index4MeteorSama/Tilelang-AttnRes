# Tilelang-AttnRes

This repository implements the [Attention Residual](https://arxiv.org/pdf/2603.15031) architecture from Kimi K3 using [TileLang](https://github.com/tile-ai/tilelang).

### 1. Speed test (on RTX 5080)

Tested with input shape `[1280, 8, 7168]` and `bfloat16` precision.

| Implementation | Latency (ms) | Speedup |
| :--- | :--- | :--- |
| **PyTorch Eager** | 4.708 | 1.00x (Baseline) |
| **Torch Compile** | 1.111 | 4.23x |
| **TileLang (Ours)** | **0.238** | **19.78x** |