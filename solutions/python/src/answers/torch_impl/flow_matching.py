"""Flow Matching 模型与采样 —— 综合面经题库-2 基础题 4 + PyTorch 编程题 3。

对应面试题:
    - 综合-2 Q4 "Flow Matching 的预测目标? 以 x0 为条件的条件速度场如何理解?"
    - 综合-2 PyTorch-3 "Flow Matching 采样伪代码"

核心公式:
    概率路径: x_t = (1-t) x_0 + t x_1,  t~U[0,1]   (线性插值; x1=数据, x0=噪声)
    条件速度场: u_t(x|x1) = x1 - x0                  (对 t 求导即得)
    边缘速度场: v_θ(x_t, t) ≈ E[x1 - x0 | x_t]      (网络回归目标)
    采样: 从 x0~N(0,I) 出发, x_{t+dt} = x_t + v_θ(x_t,t) dt (欧拉法)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class FlowMatchingLoss(nn.Module):
    """CFM(Conditional Flow Matching) 训练损失: MSE(v_θ(x_t,t), x1 - x0)。"""

    def __init__(self, sigma_min: float = 0.0) -> None:
        super().__init__()
        self.sigma_min = sigma_min

    def forward(self, net: nn.Module, x1: torch.Tensor) -> torch.Tensor:
        """x1: (B, ...) 一批真实数据样本; net(x_t, t) 预测速度场。"""
        x0 = torch.randn_like(x1)                       # 噪声起点
        t = torch.rand(x1.shape[0], device=x1.device)   # U[0,1]

        # 概率路径(线性): x_t = (1-t)x0 + t·x1 (+ sigma_min 噪声项, 常取 0)
        t_view = t.view(-1, *([1] * (x1.dim() - 1)))
        x_t = (1 - t_view) * x0 + t_view * x1
        if self.sigma_min > 0:
            x_t = x_t + self.sigma_min * torch.randn_like(x_t)

        target = x1 - x0                                # 条件速度场(与 t 无关, 线性路径)
        pred = net(x_t, t)
        return torch.nn.functional.mse_loss(pred, target)


class TinyVelocityNet(nn.Module):
    """演示用小网络: 输入 x_t 与 t 拼接, 输出同形速度。"""

    def __init__(self, dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_in = t.view(-1, 1) if t.dim() == 1 else t
        return self.net(torch.cat([x_t, t_in], dim=-1))


@torch.no_grad()
def euler_sample(net: nn.Module, shape: tuple[int, ...], num_steps: int = 40,
                 device: str = "cpu") -> torch.Tensor:
    """采样伪代码的实现版(对应面试题"编写采样过程伪代码"):

        x ~ N(0, I)
        for i in range(N):
            t = i / N
            x = x + v_θ(x, t) * (1 / N)     # 欧拉步
        return x
    面试追问: num_inference_steps=40 为何训练时 timesteps 仍要 1000?
    答: 训练是在连续/细离散 t 上回归速度场的通用算子, 与采样步数解耦;
        采样步数只影响 ODE 积分误差, 训练 timesteps 决定速度场拟合密度。
    """
    x = torch.randn(shape, device=device)
    dt = 1.0 / num_steps
    for i in range(num_steps):
        t = torch.full((shape[0],), i * dt, device=device)
        x = x + net(x, t) * dt
    return x


def train_and_sample_demo(dim: int = 2, steps: int = 500, num_steps_sample: int = 40) -> dict[str, object]:
    """端到端小实验: 让速度场学会把 N(0,I) 推到目标两个簇。"""
    torch.manual_seed(0)
    # 目标分布: 两个高斯簇
    target = torch.cat([
        torch.randn(512, dim) * 0.1 + torch.tensor([2.0, 0.0]),
        torch.randn(512, dim) * 0.1 + torch.tensor([-2.0, 0.0]),
    ])
    net = TinyVelocityNet(dim)
    loss_fn = FlowMatchingLoss()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(steps):
        opt.zero_grad()
        loss = loss_fn(net, target)
        loss.backward()
        opt.step()
    samples = euler_sample(net, (256, dim), num_steps=num_steps_sample)
    # 粗验证: 样本应聚集在 ±2 附近
    x_mean_abs = samples[:, 0].abs().mean().item()
    y_std = samples[:, 1].std().item()
    return {
        "final_loss": float(loss.detach()),
        "sample_abs_x_mean": x_mean_abs,   # 期望接近 2
        "sample_y_std": y_std,             # 期望接近 0.1
        "clustered_near_targets": bool(x_mean_abs > 1.0 and y_std < 0.6),
        # 注: 双峰目标下 loss 有不可约下限(给定中间态 x_t, x1 可来自任一簇),
        # 收敛后 loss≈1.2-1.3 是正常的, 分布指标才是正确性判据。
    }
