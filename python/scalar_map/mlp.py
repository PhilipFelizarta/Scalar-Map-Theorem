"""A small numpy MLP with the scalings from the Scalar Map Theorem.

Weights use the row-vector convention: a layer computes ``a @ W + b`` with ``W`` of shape
``(n_in, n_out)``. Everything is float64 so that invariances can be checked to ~1e-12.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Activations: value, derivative, and the scale function s (None if not a PSM)
# ---------------------------------------------------------------------------


def _gelu(z):
    return 0.5 * z * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (z + 0.044715 * z**3)))


def _gelu_grad(z):
    k = math.sqrt(2.0 / math.pi)
    u = k * (z + 0.044715 * z**3)
    t = np.tanh(u)
    return 0.5 * (1.0 + t) + 0.5 * z * (1.0 - t**2) * k * (1.0 + 3 * 0.044715 * z**2)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


@dataclass(frozen=True)
class Activation:
    name: str
    f: callable
    df: callable
    s: callable | None  # scale function of a positive scalar map, None if not a PSM

    @property
    def is_psm(self) -> bool:
        return self.s is not None


def leaky_relu(alpha: float = 0.3) -> Activation:
    return Activation(
        f"leaky_relu({alpha})",
        lambda z: np.where(z >= 0, z, alpha * z),
        lambda z: np.where(z >= 0, 1.0, alpha),
        lambda a: a,
    )


ACTIVATIONS: dict[str, Activation] = {
    "relu": Activation("relu", lambda z: np.maximum(z, 0), lambda z: (z > 0).astype(float), lambda a: a),
    "leaky_relu": leaky_relu(0.3),
    "abs": Activation("abs", np.abs, np.sign, lambda a: a),
    "linear": Activation("linear", lambda z: z, np.ones_like, lambda a: a),
    "relu2": Activation(
        "relu2", lambda z: np.maximum(z, 0) ** 2, lambda z: 2 * np.maximum(z, 0), lambda a: a**2
    ),
    "heaviside": Activation(
        "heaviside", lambda z: (z > 0).astype(float), np.zeros_like, lambda a: 1.0
    ),
    # Not positive scalar maps:
    "gelu": Activation("gelu", _gelu, _gelu_grad, None),
    "tanh": Activation("tanh", np.tanh, lambda z: 1 - np.tanh(z) ** 2, None),
    "softplus": Activation(
        "softplus", lambda z: np.logaddexp(0, z), _sigmoid, None
    ),
}


# ---------------------------------------------------------------------------
# The network
# ---------------------------------------------------------------------------


@dataclass
class MLP:
    """Hidden layers use ``acts[i]``; the last layer is affine and returns logits."""

    weights: list[np.ndarray]
    biases: list[np.ndarray]
    acts: list[Activation]
    use_bias: bool = True
    _cache: dict = field(default_factory=dict, repr=False)

    @staticmethod
    def init(sizes, act="relu", use_bias=True, seed=0) -> "MLP":
        rng = np.random.default_rng(seed)
        acts = act if isinstance(act, list) else [ACTIVATIONS[act] if isinstance(act, str) else act] * (len(sizes) - 2)
        acts = [ACTIVATIONS[a] if isinstance(a, str) else a for a in acts]
        W = [rng.normal(0, math.sqrt(2.0 / i), (i, o)) for i, o in zip(sizes[:-1], sizes[1:])]
        b = [
            (rng.normal(0, 0.1, o) if use_bias else np.zeros(o)) for o in sizes[1:]
        ]
        return MLP(W, b, acts, use_bias)

    @property
    def depth(self) -> int:
        return len(self.weights)

    def copy(self) -> "MLP":
        return MLP([w.copy() for w in self.weights], [b.copy() for b in self.biases], list(self.acts), self.use_bias)

    def logits(self, X: np.ndarray, keep=False) -> np.ndarray:
        a = X
        zs, as_ = [], [X]
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = a @ W + b
            if i == self.depth - 1:
                if keep:
                    self._cache = {"zs": zs, "as": as_}
                return z
            zs.append(z)
            a = self.acts[i].f(z)
            as_.append(a)
        raise AssertionError

    # -- training ---------------------------------------------------------
    def grads(self, X, y):
        z = self.logits(X, keep=True)
        p = softmax(z)
        p[np.arange(len(y)), y] -= 1.0
        g = p / len(y)
        dW, db = [None] * self.depth, [None] * self.depth
        zs, as_ = self._cache["zs"], self._cache["as"]
        for i in reversed(range(self.depth)):
            dW[i] = as_[i].T @ g
            db[i] = g.sum(0)
            if i > 0:
                g = (g @ self.weights[i].T) * self.acts[i - 1].df(zs[i - 1])
        return dW, db

    def flat(self) -> np.ndarray:
        parts = list(self.weights) + (list(self.biases) if self.use_bias else [])
        return np.concatenate([p.ravel() for p in parts])

    def with_flat(self, theta: np.ndarray) -> "MLP":
        out = self.copy()
        k = 0
        for lst in [out.weights] + ([out.biases] if self.use_bias else []):
            for j, p in enumerate(lst):
                lst[j] = theta[k : k + p.size].reshape(p.shape).copy()
                k += p.size
        return out


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def cross_entropy(z: np.ndarray, y: np.ndarray) -> float:
    z = z - z.max(axis=1, keepdims=True)
    lse = np.log(np.exp(z).sum(axis=1))
    return float(np.mean(lse - z[np.arange(len(y)), y]))


def topk_accuracy(z: np.ndarray, y: np.ndarray, k: int = 1) -> float:
    """Fraction of rows whose label has fewer than k strictly larger logits (as in the Lean
    definition ``InTopK``, so ties count in the label's favour)."""
    above = (z > z[np.arange(len(y)), y][:, None]).sum(axis=1)
    return float(np.mean(above < k))


def train(net: MLP, X, y, epochs=30, lr=1e-3, batch=64, seed=0, snapshots=None):
    """Adam. ``snapshots`` is an optional list of epochs after which to store a copy."""
    rng = np.random.default_rng(seed)
    params = net.weights + (net.biases if net.use_bias else [])
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    b1, b2, eps, t = 0.9, 0.999, 1e-8, 0
    snaps = {}
    for ep in range(1, epochs + 1):
        order = rng.permutation(len(y))
        for s in range(0, len(y), batch):
            idx = order[s : s + batch]
            dW, db = net.grads(X[idx], y[idx])
            gs = dW + (db if net.use_bias else [])
            t += 1
            for j, (p, g) in enumerate(zip(params, gs)):
                m[j] = b1 * m[j] + (1 - b1) * g
                v[j] = b2 * v[j] + (1 - b2) * g * g
                p -= lr * (m[j] / (1 - b1**t)) / (np.sqrt(v[j] / (1 - b2**t)) + eps)
        if snapshots and ep in snapshots:
            snaps[ep] = net.copy()
    return snaps


# ---------------------------------------------------------------------------
# The scalings from the theorems
# ---------------------------------------------------------------------------


def scale_weights(net: MLP, c) -> MLP:
    """Theorem 1: multiply layer i's weights by c[i]; biases untouched."""
    out = net.copy()
    out.weights = [ci * W for ci, W in zip(c, net.weights)]
    return out


def scale_with_bias(net: MLP, c, rule="telescoping") -> tuple[MLP, float]:
    """Theorem 2: weights by c[i], bias i by c[i] * (scale of layer i's input).

    Returns the scaled network and the predicted logit factor beta. ``rule="same"`` scales
    each bias by c[i] only (the naive rule, which breaks the theorem when c != 1).
    """
    out = net.copy()
    a = 1.0  # scale already carried by the current layer's input
    for i, ci in enumerate(c):
        out.weights[i] = ci * net.weights[i]
        out.biases[i] = (ci * a if rule == "telescoping" else ci) * net.biases[i]
        a = ci * a
        if i < net.depth - 1:
            s = net.acts[i].s
            if s is None:
                return out, float("nan")
            a = s(a)
    return out, a


def predicted_factor(net: MLP, c) -> float:
    return scale_with_bias(net, c)[1]


def rescale_neurons(net: MLP, layer: int, d: np.ndarray) -> MLP:
    """Per-neuron symmetry (degree-1 activations): multiply hidden unit j of ``layer`` by
    d[j] on the way in (weights and bias) and by 1/d[j] on the way out."""
    out = net.copy()
    out.weights[layer] = net.weights[layer] * d[None, :]
    out.biases[layer] = net.biases[layer] * d
    out.weights[layer + 1] = net.weights[layer + 1] / d[:, None]
    return out


# ---------------------------------------------------------------------------
# Cross-entropy along a ray: convex in the logit factor t, solved exactly
# ---------------------------------------------------------------------------


def ce_along_ray(z: np.ndarray, y: np.ndarray, t: float) -> float:
    return cross_entropy(t * z, y)


def best_logit_scale(z: np.ndarray, y: np.ndarray, t0: float = 1.0, tol=1e-12, max_iter=100):
    """Minimise L(t) = mean(logsumexp(t z) - t z_y) over t > 0 by safeguarded Newton.

    L'(t)  = mean(E_p[z] - z_y)      (p = softmax(t z))
    L''(t) = mean(Var_p[z]) >= 0      so L is convex and the minimiser is unique when it exists
    (it does unless the data are separated, in which case L decreases forever).
    """
    zy = z[np.arange(len(y)), y]
    t = t0
    for _ in range(max_iter):
        p = softmax(t * z)
        mean = (p * z).sum(1)
        var = (p * z * z).sum(1) - mean**2
        g = float(np.mean(mean - zy))
        h = float(np.mean(var))
        if h <= 0:
            break
        step = g / h
        t_new = t - step
        if t_new <= 0:
            t_new = t / 2
        if abs(t_new - t) < tol * max(1.0, t):
            t = t_new
            break
        t = t_new
    return t
