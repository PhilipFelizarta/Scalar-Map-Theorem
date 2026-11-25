"""Experiment 1: the theorems on trained networks.

Trains MLPs on digits, applies 200 random per-layer scalings (log-uniform in [0.1, 10]) and
records how far the test accuracy and the logits move. Expected: exactly 0 change for PSM
activations with the theorem's bias rule; visible change for the naive bias rule, for the
uniform ray alpha * theta on a network with biases, and for GELU / tanh / softplus.
"""
import numpy as np
from common import save_json

from scalar_map import (
    MLP, digits_splits, predicted_factor, rescale_neurons, scale_weights, scale_with_bias,
    topk_accuracy, train,
)

(Xtr, ytr), (Xva, yva), (Xte, yte) = digits_splits()
SIZES = [64, 100, 100, 100, 10]
rng = np.random.default_rng(0)
C = np.exp(rng.uniform(np.log(0.1), np.log(10), (200, len(SIZES) - 1)))


def acc_range(nets_logits):
    accs = [topk_accuracy(z, yte, 1) for z in nets_logits]
    top3 = [topk_accuracy(z, yte, 3) for z in nets_logits]
    return float(max(accs) - min(accs)), float(max(top3) - min(top3))


rows = []
CASES = [
    ("leaky_relu(0.3), no bias (the notebook)", "leaky_relu", False, "theorem 1"),
    ("ReLU, no bias", "relu", False, "theorem 1"),
    ("ReLU, biases, telescoping rule", "relu", True, "theorem 2"),
    ("ReLU, biases, naive rule (bias i by c_i)", "relu", True, "naive"),
    ("ReLU, biases, uniform ray alpha*theta", "relu", True, "ray"),
    ("ReLU^2, biases, telescoping rule", "relu2", True, "theorem 2"),
    ("GELU, no bias", "gelu", False, "weights"),
    ("tanh, no bias", "tanh", False, "weights"),
    ("softplus, no bias", "softplus", False, "weights"),
]
trained = {}
for label, act, bias, rule in CASES:
    key = (act, bias)
    if key not in trained:
        net = MLP.init(SIZES, act, use_bias=bias, seed=0)
        lr = 3e-4 if act == "relu2" else 1e-3
        train(net, Xtr, ytr, epochs=40, lr=lr, seed=0)
        trained[key] = net
    net = trained[key]
    z0 = net.logits(Xte)
    zs, rel = [z0], []
    for c in C:
        if rule == "theorem 1":
            z = scale_weights(net, c).logits(Xte)
            beta = predicted_factor(net, c)
        elif rule == "theorem 2":
            s, beta = scale_with_bias(net, c)
            z = s.logits(Xte)
        elif rule == "naive":
            z = scale_with_bias(net, c, rule="same")[0].logits(Xte)
            beta = np.nan
        elif rule == "ray":
            a = c[0]
            z = net.with_flat(a * net.flat()).logits(Xte)
            beta = np.nan
        else:
            z = scale_weights(net, c).logits(Xte)
            beta = np.nan
        zs.append(z)
        if np.isfinite(beta):
            rel.append(float(np.max(np.abs(z - beta * z0)) / np.max(np.abs(beta * z0))))
    d1, d3 = acc_range(zs)
    rows.append({
        "case": label,
        "test_acc": topk_accuracy(z0, yte, 1),
        "top1_range": d1,
        "top3_range": d3,
        "max_rel_logit_error_vs_beta": max(rel) if rel else None,
    })
    print(rows[-1])

# Per-neuron symmetry on the biased ReLU net: identical logits, not just identical accuracy.
net = trained[("relu", True)]
out = net
for layer in range(net.depth - 1):
    out = rescale_neurons(out, layer, np.exp(rng.uniform(np.log(0.1), np.log(10), SIZES[layer + 1])))
neuron_err = float(np.max(np.abs(out.logits(Xte) - net.logits(Xte))))
print("per-neuron rescaling, max |logit change|:", neuron_err)

save_json("exp1_invariance", {
    "n_scalings": len(C), "scaling_range": [0.1, 10], "n_test": int(len(yte)),
    "rows": rows, "per_neuron_rescaling_max_abs_logit_change": neuron_err,
    "free_positive_parameters_per_layer_vs_per_neuron": {
        "per_layer": len(SIZES) - 1, "per_neuron_plus_scale": int(sum(SIZES[1:-1]) + 1)},
})
