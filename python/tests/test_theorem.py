"""Property tests for the theorems proved in ../lean (random networks, no training)."""
import numpy as np
import pytest

from scalar_map import (
    ACTIVATIONS, MLP, best_logit_scale, cross_entropy, leaky_relu, predicted_factor,
    rescale_neurons, scale_weights, scale_with_bias, topk_accuracy,
)

RNG = np.random.default_rng(1)
X = RNG.normal(size=(500, 16))
SIZES = [16, 32, 32, 24, 10]
PSMS = ["relu", "leaky_relu", "abs", "linear", "relu2", "heaviside"]


def log_uniform(n, lo=1e-2, hi=1e2, rng=RNG):
    return np.exp(rng.uniform(np.log(lo), np.log(hi), n))


@pytest.mark.parametrize("act", PSMS)
@pytest.mark.parametrize("seed", range(5))
def test_theorem1_bias_free_logits_scale_by_one_factor(act, seed):
    net = MLP.init(SIZES, act, use_bias=False, seed=seed)
    c = log_uniform(net.depth)
    beta = predicted_factor(net, c)
    z, zs = net.logits(X), scale_weights(net, c).logits(X)
    np.testing.assert_allclose(zs, beta * z, rtol=1e-9, atol=1e-12 * abs(beta))
    for k in (1, 3, 5):
        assert topk_accuracy(zs, Y, k) == topk_accuracy(z, Y, k)


Y = RNG.integers(0, 10, len(X))


@pytest.mark.parametrize("act", PSMS)
@pytest.mark.parametrize("seed", range(5))
def test_theorem2_biases_with_telescoping_rule(act, seed):
    net = MLP.init(SIZES, act, use_bias=True, seed=seed)
    c = log_uniform(net.depth)
    scaled, beta = scale_with_bias(net, c)
    np.testing.assert_allclose(scaled.logits(X), beta * net.logits(X), rtol=1e-9, atol=1e-12 * abs(beta))


def test_degree1_factor_is_product():
    net = MLP.init(SIZES, "leaky_relu", use_bias=False, seed=0)
    c = log_uniform(net.depth)
    assert np.isclose(predicted_factor(net, c), np.prod(c))


def test_relu2_factor_telescopes():
    net = MLP.init([16, 8, 8, 10], "relu2", use_bias=False)
    c = [2.0, 3.0, 5.0]
    # s(a) = a^2:  beta = c2 * (c1 * c0^2)^2
    assert np.isclose(predicted_factor(net, c), 5.0 * (3.0 * 2.0**2) ** 2)


def test_naive_bias_rule_breaks():
    net = MLP.init(SIZES, "relu", use_bias=True, seed=0)
    c = [3.0, 0.5, 4.0, 2.0]
    naive, _ = scale_with_bias(net, c, rule="same")
    _, beta = scale_with_bias(net, c)
    assert not np.allclose(naive.logits(X), beta * net.logits(X), rtol=1e-3)


@pytest.mark.parametrize("act", ["gelu", "tanh", "softplus"])
def test_non_psm_activations_break(act):
    net = MLP.init(SIZES, act, use_bias=False, seed=0)
    z = net.logits(X)
    zs = scale_weights(net, [3.0] * net.depth).logits(X)
    beta = np.sum(zs * z) / np.sum(z * z)  # best single factor
    assert np.linalg.norm(zs - beta * z) > 1e-3 * np.linalg.norm(zs)


def test_leaky_relu_is_not_odd_so_negative_scalings_break():
    net = MLP.init(SIZES, leaky_relu(0.3), use_bias=False, seed=0)
    z = net.logits(X)
    zn = scale_weights(net, [-1.0, 1.0, 1.0, 1.0]).logits(X)
    assert not np.allclose(zn, z) and not np.allclose(zn, -z)


@pytest.mark.parametrize("act", ["relu", "leaky_relu", "abs"])
def test_per_neuron_rescaling_is_exact(act):
    net = MLP.init(SIZES, act, use_bias=True, seed=0)
    out = net
    for layer in range(net.depth - 1):
        out = rescale_neurons(out, layer, log_uniform(SIZES[layer + 1]))
    np.testing.assert_allclose(out.logits(X), net.logits(X), rtol=1e-9, atol=1e-10)


def test_temperature_newton_matches_dense_grid():
    net = MLP.init(SIZES, "relu", use_bias=False, seed=3)
    z = net.logits(X) / 10
    y = z.argmax(1)
    flip = RNG.random(len(y)) < 0.3
    y = np.where(flip, RNG.integers(0, 10, len(y)), y)
    t = best_logit_scale(z, y)
    grid = np.linspace(0.01, 10 * t, 20001)
    losses = [cross_entropy(g * z, y) for g in grid]
    assert cross_entropy(t * z, y) <= min(losses) + 1e-9
    # convexity along the ray: second differences are non-negative
    assert np.all(np.diff(losses, 2) > -1e-12)
