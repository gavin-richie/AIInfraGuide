"""MLA (Multi-head Latent Attention, DeepSeek-V2/V3) 手写实现。

对应面试题:
    - 蚂蚁-校招-一面 "MLA 优势、权重吸收问题"
    - 联想 "手写 MHA 与 Mask 位置" / 小厂 "MLA prefill/decode 复杂度与矩阵吸收"
    - 快手-一面-3 "MLA decode 计算访存比"(理论, 见文档 04)

核心思想:
    1. 低秩压缩: h_t → c^KV_t = W_DKV h_t (d_c 比n_heads*d_head小得多)
    2. KV Cache 只存 c^KV_t (以及 RoPE 部分的 c^K_R_t), 解压在用时分摊
    3. 权重吸收: attention 里的 W_UK/W_UV 可以和 W_Q/W_O 合并,
       decode 时直接用 c^KV 算, 省去显式解压 —— 这就是"矩阵吸收"考点
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadLatentAttention(nn.Module):
    """简化版 MLA: 压缩 → 升维 → attention, 支持吸收与不吸收两种前向。

    记号(与 DeepSeek-V2 论文一致):
        d_c      : KV 压缩维度(latent dim, 如 512)
        d_c'     : Q 压缩维度(如 1536)
        n_h*d_h  : 多头总维度
        n_l      : RoPE 部分每头维度(如 64)

    为突出考点本实现省去 RoPE 拼接分支(absorb 推导同理), 完整版见文档 04。
    """

    def __init__(self, d_model: int, n_heads: int, d_head: int, d_latent: int) -> None:
        super().__init__()
        self.n_heads, self.d_head, self.d_latent = n_heads, d_head, d_latent
        self.scaling = 1.0 / math.sqrt(d_head)
        d_attn = n_heads * d_head

        self.q_down = nn.Linear(d_model, d_latent, bias=False)   # W_DQ
        self.q_up = nn.Linear(d_latent, d_attn, bias=False)      # W_UQ
        self.kv_down = nn.Linear(d_model, d_latent, bias=False)  # W_DKV  (唯一需要缓存的投影)
        self.k_up = nn.Linear(d_latent, d_attn, bias=False)      # W_UK
        self.v_up = nn.Linear(d_latent, d_attn, bias=False)      # W_UV
        self.o_proj = nn.Linear(d_attn, d_model, bias=False)     # W_O

    # ---------------------------------------------------------- 朴素前向(便于理解)
    def forward_naive(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        """x: (B, S, E)。KV Cache 存 kv_down(x) 即 (B, S, d_latent)。"""
        bsz, seq, _ = x.shape
        h = self.n_heads

        q = self.q_up(self.q_down(x))          # (B,S,d_attn)
        c_kv = self.kv_down(x)                  # (B,S,d_latent) ← 缓存这个
        k = self.k_up(c_kv)                     # (B,S,d_attn)
        v = self.v_up(c_kv)

        q = q.view(bsz, seq, h, self.d_head).transpose(1, 2)
        k = k.view(bsz, seq, h, self.d_head).transpose(1, 2)
        v = v.view(bsz, seq, h, self.d_head).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) * self.scaling
        if causal:
            mask = torch.triu(torch.ones(seq, seq, dtype=torch.bool, device=x.device), 1)
            scores = scores.masked_fill(mask, float("-inf"))
        out = torch.softmax(scores, dim=-1) @ v
        out = out.transpose(1, 2).contiguous().view(bsz, seq, -1)
        return self.o_proj(out)

    # ---------------------------------------------------------- 权重吸收前向
    def forward_absorb(self, cached_kv: torch.Tensor, x_new: torch.Tensor) -> torch.Tensor:
        """decode 路径: 把 W_UK/W_UV 吸收进 W_UQ/W_O, 直接在 latent 空间算。

        推导:
            q_h = W_UQ c^Q            (B,S,h*d)
            k_h = W_UK c^KV
            score = q_h k_h^T = (W_UQ c^Q)(W_UK c^KV)^T
                  = c^Q (W_UQ^T W_UK) c^KV^T          ← W_UK 被吸收成 (d_latent,d_latent)
            同理 v 侧: attn @ (W_UV c^KV) 可以先算 attn @ c^KV 再升维(W_UV 吸收进 W_O)
        好处: decode 每步对 KV 的访存从 n_h*d_h 降到 d_latent, 且免去解压 kernel。
        cached_kv: (B, S_total, d_latent) 历史压缩 KV; x_new: (B, 1, E) 当前 token。
        """
        bsz, _, _ = x_new.shape
        h, d = self.n_heads, self.d_head

        c_q = self.q_down(x_new)                                  # (B,1,d_latent)
        q = self.q_up(c_q).view(bsz, 1, h, d).transpose(1, 2)     # (B,h,1,d)

        # 吸收后的逐头"latent 空间"投影: W_UQ_h^T @ W_UK_h → (d_latent, d_latent)
        # k_up.weight: (d_attn, d_latent) 行优先按头切
        w_uk = self.k_up.weight.view(h, d, -1)                    # (h, d, d_latent)
        w_uv = self.v_up.weight.view(h, d, -1)                    # (h, d, d_latent)
        w_o = self.o_proj.weight.view(-1, h, d)                   # (d_model, h, d)

        # score = q (W_uk^T c_kv^T): 先 latent 再投影, 顺序可换(结合律)
        k_latent = torch.einsum("hdl,bcl->bhdc", w_uk, cached_kv)  # (B,h,d_latent,d)
        scores = torch.einsum("bhd,bhdc->bhc", q.squeeze(2).float(), k_latent.float())
        scores = scores * self.scaling                             # (B,h,S)
        attn = torch.softmax(scores, dim=-1)

        # v 侧先在 latent 空间加权求和, 再经 (W_UV 吸收 W_O) 升维输出
        summed_latent = torch.einsum("bhs,bsl->bhl", attn, cached_kv)  # (B,h,d_latent)
        # o = W_O W_UV (summed per head): 等价 einsum
        out = torch.einsum("bhl,hdl,ehd->be", summed_latent.float(), w_uv.float(), w_o.float())
        return out.unsqueeze(1)


@torch.no_grad()
def absorb_equivalence_demo() -> dict[str, object]:
    """验证 absorb 前向与 naive 前向数值等价(单 token decode 场景)。"""
    torch.manual_seed(0)
    mla = MultiHeadLatentAttention(d_model=64, n_heads=4, d_head=16, d_latent=32)
    mla.eval()
    x = torch.randn(1, 8, 64)
    cached = mla.kv_down(x)            # 模拟已缓存的前 8 个 token 的 latent KV
    x_new = x[:, -1:, :]               # 第 8 个 token 作为当前步输入
    naive = mla.forward_naive(x_new[:, :1, :], causal=False)  # 单 token 无 mask
    # naive 的 KV 来自 x_new 自身; absorb 版需要全部历史 → 用同权重重算
    q = mla.q_up(mla.q_down(x_new))
    c_all = cached
    k = mla.k_up(c_all).view(1, 8, 4, 16).transpose(1, 2)
    v = mla.v_up(c_all).view(1, 8, 4, 16).transpose(1, 2)
    qq = q.view(1, 1, 4, 16).transpose(1, 2)
    ref = torch.nn.functional.scaled_dot_product_attention(qq, k, v)
    ref = mla.o_proj(ref.transpose(1, 2).contiguous().view(1, 1, 64))
    got = mla.forward_absorb(c_all, x_new)
    return {
        "max_abs_err": float((ref - got).abs().max()),
        "allclose": bool(torch.allclose(ref, got, atol=1e-4)),
    }
