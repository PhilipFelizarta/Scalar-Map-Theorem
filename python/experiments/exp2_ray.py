"""Experiment 2: along a ray, accuracy is flat and cross-entropy is convex in t = alpha^L.

Picks the scale on validation by Newton's method (exact) and compares with the notebook's
400-point grid search.
"""
import numpy as np
from common import BLUE, INK2, ORANGE, plt, save_fig, save_json

from scalar_map import MLP, best_logit_scale, cross_entropy, digits_splits, topk_accuracy, train

(Xtr, ytr), (Xva, yva), (Xte, yte) = digits_splits()
net = MLP.init([64, 100, 100, 100, 10], "leaky_relu", use_bias=False, seed=0)
train(net, Xtr, ytr, epochs=40, seed=0)
L = net.depth
zva, zte, ztr = net.logits(Xva), net.logits(Xte), net.logits(Xtr)

alphas = np.linspace(0.3, 1.5, 241)
rows = {k: [] for k in ("train_ce", "val_ce", "test_ce", "val_acc", "test_acc")}
for a in alphas:
    t = a**L  # Theorem 1, degree 1: logits scale by alpha^L
    rows["train_ce"].append(cross_entropy(t * ztr, ytr))
    rows["val_ce"].append(cross_entropy(t * zva, yva))
    rows["test_ce"].append(cross_entropy(t * zte, yte))
    rows["val_acc"].append(topk_accuracy(t * zva, yva))
    rows["test_acc"].append(topk_accuracy(t * zte, yte))

t_star = best_logit_scale(zva, yva)
a_star = t_star ** (1 / L)
grid = np.linspace(0.01, 4, 400)  # the notebook's grid search, done on validation
a_grid = grid[int(np.argmin([cross_entropy(g**L * zva, yva) for g in grid]))]
res = {
    "depth": L, "alpha_newton": a_star, "alpha_grid400": float(a_grid),
    "val_ce_at_1": cross_entropy(zva, yva), "val_ce_newton": cross_entropy(t_star * zva, yva),
    "val_ce_grid400": cross_entropy(a_grid**L * zva, yva),
    "test_ce_at_1": cross_entropy(zte, yte), "test_ce_newton": cross_entropy(t_star * zte, yte),
    "test_acc_at_1": topk_accuracy(zte, yte), "test_acc_newton": topk_accuracy(t_star * zte, yte),
    "accuracy_values_seen_along_ray": sorted(set(rows["test_acc"])),
    # Convex in t = alpha^L (not in alpha): check second differences on a uniform t grid.
    "ce_second_differences_in_t_min": float(np.min(np.diff(
        [cross_entropy(t * zva, yva) for t in np.linspace(0.05, 4, 400)], 2))),
}
print(res)
save_json("exp2_ray", res)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.4))
ax1.plot(alphas, rows["val_ce"], color=BLUE, label="validation")
ax1.plot(alphas, rows["test_ce"], color=ORANGE, label="test")
ax1.axvline(a_star, color=INK2, lw=1, ls="--")
ax1.annotate(f"Newton on validation\nalpha* = {a_star:.3f}", (a_star, max(rows["val_ce"]) * 0.8),
             xytext=(8, 0), textcoords="offset points", color=INK2, fontsize=9)
ax1.set(xlabel="alpha (every weight multiplied by alpha)", ylabel="cross-entropy",
        title="Loss changes along the ray")
ax1.legend()
ax2.plot(alphas, rows["val_acc"], color=BLUE, label="validation")
ax2.plot(alphas, rows["test_acc"], color=ORANGE, label="test")
ax2.set(xlabel="alpha", ylabel="top-1 accuracy", title="Accuracy does not",
        ylim=(min(rows["test_acc"] + rows["val_acc"]) - 0.02, 1.0))
ax2.legend(loc="lower right")
save_fig(fig, "exp2_ray")
