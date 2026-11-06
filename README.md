# Tilelang-AttnRes(forward only)

This repository implements the [Attention Residual](https://arxiv.org/pdf/2603.15031) architecture from Kimi K3 using [TileLang](https://github.com/tile-ai/tilelang).

### 1. Benchmark (RTX 5080, Windows 11, bf16)

Tested with input shape `[1280, 8, 7168]` and `bfloat16` precision.

| Shape (num_tokens x num_blocks x hidden_size) | Implementation | Latency (ms) / Speedup |
| :--- | :--- | :--- |
| **1280 x 8 x 7168** | PyTorch Eager (Baseline) | 4.694 / 1.00x |
| | Torch Compile | 1.110 / 4.23x |
| | **TileLang** | **0.248 / 18.93x** |
| **15360 x 8 x 768** | PyTorch Eager (Baseline) | 6.146 / 1.00x |
| | Torch Compile | 1.822 / 3.37x |
| | **TileLang** | **0.357 / 17.19x** |
| **768 x 1 x 4096** | PyTorch Eager (Baseline) | 0.231 / 1.00x |
| | Torch Compile | 0.082 / 2.80x |
| | **TileLang** | **0.034 / 6.64x** |

### 2. Usage

```python
import torch
from attn_res_fwd import block_residual_fwd

# block_residual: (num_tokens, num_blocks, hidden_size)
# prefix_sum:     (num_tokens, hidden_size)
# proj_w, norm_w: (hidden_size,)
num_tokens, num_blocks, hidden_size = 1280, 8, 7168
dtype = torch.bfloat16
device = "cuda"

prefix_sum = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
block_residual = torch.randn(num_tokens, num_blocks, hidden_size, dtype=dtype, device=device)

proj_w = torch.randn(hidden_size, dtype=torch.float32, device=device)
norm_w = torch.randn(hidden_size, dtype=torch.float32, device=device)

# forward
output, rsqrt, max_idx, exp_out, div_out, softmax_out = block_residual_fwd(
    block_residual, prefix_sum, proj_w, norm_w, eps=1e-6
)

# output:      (num_tokens, hidden_size)          bf16 --- result ---

# rsqrt:       (num_tokens, num_blocks + 1)       fp32 --- saved for backward ---
# max_idx:     (num_tokens, 1)                    int64
# exp_out:     (num_tokens, num_blocks + 1)       fp32
# div_out:     (num_tokens, 1)                    fp32
# softmax_out: (num_tokens, num_blocks + 1)       fp32
```

### 3. Future
Implement backward

### 4. Citation
```bib
@misc{chen2026attnres,
  title         = {Attention Residuals},
  author        = {Kimi Team  and Chen, Guangyu  and Zhang, Yu  and Su, Jianlin  and Xu, Weixin  and Pan, Siyuan  and Wang, Yaoyu  and Wang, Yucheng  and Chen, Guanduo  and Yin, Bohong  and Chen, Yutian  and Yan, Junjie  and Wei, Ming  and Zhang, Y.  and Meng, Fanqing  and Hong, Chao  and Xie, Xiaotong  and Liu, Shaowei  and Lu, Enzhe  and Tai, Yunpeng  and Chen, Yanru  and Men, Xin  and Guo, Haiqing  and Charles, Y.  and Lu, Haoyu  and Sui, Lin  and Zhu, Jinguo  and Zhou, Zaida  and He, Weiran  and Huang, Weixiao  and Xu, Xinran  and Wang, Yuzhi  and Lai, Guokun  and Du, Yulun  and Wu, Yuxin  and Yang, Zhilin  and Zhou, Xinyu},
  year          = {2026},
  archiveprefix = {arXiv},
  eprint        = {2603.15031},
  primaryclass  = {cs.CL}
}
```