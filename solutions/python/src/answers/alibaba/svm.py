"""手写线性 SVM（阿里巴巴-AI-Infra-实习-一二面 二面 Q6）。

对应面试题："手写 SVM 的核心实现。"——核心 = hinge loss + (次)梯度法。

原理：线性 SVM 求解
    min_{w,b}  (1/2)||w||^2 + C · Σ max(0, 1 - y_i(w·x_i + b))
合页损失 hinge(z) = max(0, 1-z) 处处连续、z=1 处不可导 → 次梯度法：
    ∇_w L = w - C · Σ_{margin<1} y_i x_i     （对满足 1 - y f(x) > 0 的样本）

工程化版本（Pegasos 风格）：每次抽 1 个（或一小批）样本，
    w ← (1 - η λ) w + η · y x   （若该样本违界 y·(w·x+b) < 1，否则只衰减 w）
学习率 η = 1/(λ t) 随步数衰减，理论上有 O(1/ε) 收敛保证。

SVM 的对偶/SMO/核技巧是标准追问（见模块尾注），手写题答到
"primal + hinge + 次梯度/PEGASOS" 即为完整。
"""
from __future__ import annotations

import numpy as np


class LinearSVM:
    """次梯度下降（Pegasos 风格）训练的线性 SVM，支持二分类 y ∈ {+1,-1}。"""

    def __init__(self, n_features: int, lam: float = 1e-3, seed: int = 0):
        self.lam = lam            # 正则系数 λ（= 1/C 视角）
        self.w = np.zeros(n_features, dtype=np.float64)
        self.b = 0.0
        self._rng = np.random.default_rng(seed)
        self._t = 0

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return X @ self.w + self.b

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.where(self.decision_function(X) >= 0.0, 1, -1)

    def hinge_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        margin = y * self.decision_function(X)
        return float(np.maximum(0.0, 1.0 - margin).mean())

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 50,
            batch_size: int = 16, verbose: bool = False) -> list[float]:
        """小批次次梯度下降。返回每 epoch 的平均 hinge loss。"""
        n = X.shape[0]
        losses: list[float] = []
        for _ in range(epochs):
            idx = self._rng.permutation(n)
            for start in range(0, n, batch_size):
                self._t += 1
                eta = 1.0 / (self.lam * self._t)          # Pegasos 学习率
                batch = idx[start:start + batch_size]
                xb, yb = X[batch], y[batch]
                margins = yb * (xb @ self.w + self.b)
                violate = margins < 1.0                    # 违界样本（次梯度激活）
                # w ← (1-ηλ)w + (η/|B|) Σ_{violate} y_i x_i
                self.w *= 1.0 - eta * self.lam
                if violate.any():
                    self.w += eta * (yb[violate, None] * xb[violate]).mean(axis=0) * batch_size
                    self.b += eta * yb[violate].mean() * batch_size
            losses.append(self.hinge_loss(X, y))
        return losses

    def fit_fullbatch(self, X: np.ndarray, y: np.ndarray, lr: float = 0.1,
                      epochs: int = 2000, C: float = 1.0) -> list[float]:
        """全样本次梯度版（教学对照：直接对 (1/2)||w||^2 + C·Σhinge 求导）。"""
        losses: list[float] = []
        for _ in range(epochs):
            margins = y * (X @ self.w + self.b)
            violate = margins < 1.0
            grad_w = self.w - C * (y[violate, None] * X[violate]).sum(axis=0)
            grad_b = -C * y[violate].sum()
            self.w -= lr * grad_w / len(y)
            self.b -= lr * grad_b / len(y)
            losses.append(self.hinge_loss(X, y))
        return losses
