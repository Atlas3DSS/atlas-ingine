#!/usr/bin/env python3
"""Your first Plato's Key in 40 lines of plain NumPy (runs on any CPU in <1 second).

Idea:
  - Model A (your local LLM) thinks about 640 prompts and produces vectors X.
  - Model B (a giant 27 GB image/video text encoder) thinks about the same prompts
    and produces vectors Y.
  - Instead of running Model B ever again, solve ONE exact math equation (ridge
    regression) in FP64 to get a small translator matrix R ("Plato's Key").
  - Also fit a "shuffled control" (B0) on mismatched prompts to prove the key
    actually learned real meaning instead of guessing the average.
"""
from __future__ import annotations
import numpy as np


def fit_platos_key(X: np.ndarray, Y: np.ndarray, lam: float = 100.0):
    """Closed-form FP64 centered ridge map: y_hat = mu_Y + (x - mu_X) @ R."""
    X64 = X.astype(np.float64)
    Y64 = Y.astype(np.float64)
    mu_X = X64.mean(axis=0, keepdims=True)
    mu_Y = Y64.mean(axis=0, keepdims=True)
    Xc = X64 - mu_X
    Yc = Y64 - mu_Y
    d = Xc.shape[1]
    R = np.linalg.solve(Xc.T @ Xc + lam * np.eye(d), Xc.T @ Yc)
    return R, mu_X, mu_Y


def apply_platos_key(X: np.ndarray, R: np.ndarray, mu_X: np.ndarray, mu_Y: np.ndarray):
    return mu_Y + (X.astype(np.float64) - mu_X) @ R


def mean_cosine(A: np.ndarray, B: np.ndarray) -> float:
    dot = np.sum(A * B, axis=1)
    denom = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-12
    return float(np.mean(dot / denom))


def main() -> None:
    rng = np.random.default_rng(42)

    # 1. Pretend we have 640 training prompts + 160 unseen test prompts.
    #    Each prompt lives in a shared semantic world, seen through Model A (d=128)
    #    and Model B (d=256) with different rotations, scales, and small noise.
    n_train, n_test, d_world, d_a, d_b = 640, 160, 64, 128, 256
    Z_train = rng.standard_normal((n_train, d_world))
    Z_test = rng.standard_normal((n_test, d_world))
    W_a = rng.standard_normal((d_world, d_a))
    W_b = rng.standard_normal((d_world, d_b)) * 5.0  # anisotropic target scale

    X_train = Z_train @ W_a + 0.15 * rng.standard_normal((n_train, d_a))
    Y_train = Z_train @ W_b + 0.15 * rng.standard_normal((n_train, d_b)) + 3.0
    X_test = Z_test @ W_a + 0.15 * rng.standard_normal((n_test, d_a))
    Y_test = Z_test @ W_b + 0.15 * rng.standard_normal((n_test, d_b)) + 3.0

    # 2. Fit the real Plato's Key (0 gradient steps, 0 GPU hours).
    R, mu_X, mu_Y = fit_platos_key(X_train, Y_train, lam=10.0)
    Y_pred = apply_platos_key(X_test, R, mu_X, mu_Y)

    # 3. Fit the shuffled control (B0) on scrambled prompt pairs.
    perm = rng.permutation(n_train)
    R_b0, mu_Xb, mu_Yb = fit_platos_key(X_train[perm], Y_train, lam=10.0)
    Y_b0 = apply_platos_key(X_test, R_b0, mu_Xb, mu_Yb)

    cos_key = mean_cosine(Y_pred, Y_test)
    cos_b0 = mean_cosine(Y_b0, Y_test)
    print(f"Real Plato's Key cosine on 160 unseen prompts:     {cos_key:.4f}")
    print(f"Shuffled Control (B0) cosine on 160 unseen prompts:{cos_b0:.4f}")
    assert cos_key > 0.95 and cos_key > cos_b0 + 0.50
    print("PASS: Your closed-form Plato's Key translated Model A -> Model B with 0 training steps!")


if __name__ == "__main__":
    main()
