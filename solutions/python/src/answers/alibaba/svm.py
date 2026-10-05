"""手写线性 SVM 核心实现（阿里巴巴-AI-Infra-实习-一二面 二面 Q6）。

核心三件套：
1. hinge loss：L = C · mean(max(0, 1 - y·(w·x+b))) + 0.5·||w||²（标签 y∈{-1,+1}）；
2. 子梯度下降（Pegasos 风格）：margin ≥ 1 的样本只有正则项梯度 w，
   margin < 1 的样本额外贡献 -y·x / -y（hinge 的次梯度）；
3. 预测 sign(w·x + b)；支持间隔 decision_function 与核技巧 rbf_kernel
   （线性不可分时预先计算 Gram 矩阵再做对偶训练，本文件给出核函数与演示）。
"""

from __future__ import annotations

import torch


def hinge_loss(w: torch.Tensor, b: torch.Tensor, X: torch.Tensor, y: torch.Tensor, C: float = 1.0) -> torch.Tensor:
    """软间隔 SVM 的经验损失（未除样本数版见 fit 内注释）。"""
    margins = y * (X @ w + b)
    return C * torch.clamp(1.0 - margins, min=0.0).mean() + 0.5 * w @ w


class LinearSVM:
    """子梯度下降训练的线性 SVM（标签 y ∈ {-1, +1}）。"""

    def __init__(self, n_features: int, C: float = 1.0, lr: float = 0.1, epochs: int = 200, seed: int = 0) -> None:
        self.C = C
        self.lr = lr
        self.epochs = epochs
        gen = torch.Generator().manual_seed(seed)
        self.w = torch.randn(n_features, generator=gen) * 0.01
        self.b = torch.zeros(1)

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> "LinearSVM":
        n = X.shape[0]
        for _ in range(self.epochs):
            margins = y * (X @ self.w + self.b)
            viol = margins < 1.0  # 违反间隔的样本，hinge 次梯度非零
            # dL/dw = w - (C/n) * Σ_{viol} y_i x_i；dL/db = -(C/n) * Σ_{viol} y_i
            grad_w = self.w - (self.C / n) * ((X * y.unsqueeze(1))[viol]).sum(dim=0)
            grad_b = -(self.C / n) * y[viol].sum()
            self.w = self.w - self.lr * grad_w
            self.b = self.b - self.lr * grad_b
        return self

    def decision_function(self, X: torch.Tensor) -> torch.Tensor:
        return X @ self.w + self.b

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        return torch.where(self.decision_function(X) >= 0, 1.0, -1.0)

    def accuracy(self, X: torch.Tensor, y: torch.Tensor) -> float:
        return (self.predict(X) == y).float().mean().item()


def rbf_kernel(X: torch.Tensor, Y: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    """RBF 核 K(x,y) = exp(-gamma ||x-y||²)。Gram 矩阵用于核 SVM 对偶训练。"""
    d2 = torch.cdist(X, Y) ** 2
    return torch.exp(-gamma * d2)


def make_separable_blobs(n_per_class: int = 64, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """构造线性可分的两簇数据（固定种子，测试可复现）。"""
    gen = torch.Generator().manual_seed(seed)
    x_pos = torch.randn(n_per_class, 2, generator=gen) * 0.4 + torch.tensor([2.0, 2.0])
    x_neg = torch.randn(n_per_class, 2, generator=gen) * 0.4 + torch.tensor([-2.0, -2.0])
    X = torch.cat([x_pos, x_neg])
    y = torch.cat([torch.ones(n_per_class), -torch.ones(n_per_class)])
    return X, y
