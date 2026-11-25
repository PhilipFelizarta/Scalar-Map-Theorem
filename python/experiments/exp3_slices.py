"""Experiment 3: the notebook's loss-landscape slices, with a clean protocol.

The notebook mixes three weight vectors w0 (init), m1, m2 (snapshots) as
x*w0 + y*m1 + z*m2 with (x, y, z) on the unit sphere. For a bias-free degree-1 network,
Theorem 1 says every point of the 3-D span, except 0, has the same predictions as its
projection onto that sphere, so the sphere covers the whole span. (Not the half-sphere:
leaky ReLU is not odd, so -theta is a different network.)

The notebook picked the best point on test data and reported test accuracy there. Here the
point is picked on validation, and test is looked at once. The gap between "best on test" and
"best on validation, scored on test" is the optimism of the old protocol.
"""
import numpy as np
from common import AQUA, BLUE, INK2, ORANGE, SEQ, plt, save_fig, save_json
from matplotlib.colors import LinearSegmentedColormap

from scalar_map import MLP, cross_entropy, digits_splits, topk_accuracy, train

(Xtr, ytr), (Xva, yva), (Xte, yte) = digits_splits()
SIZES = [64, 100, 100, 100, 10]
NT, NP = 72, 36  # theta in [0, 2pi), phi in [0, pi]
thetas = np.arange(NT) / NT * 2 * np.pi
phis = (np.arange(NP) + 0.5) / NP * np.pi


def sphere_scan(net, w0, w1, w2):
    acc_va, acc_te, ce_va = (np.zeros((NT, NP)) for _ in range(3))
    for i, th in enumerate(thetas):
        for j, ph in enumerate(phis):
            x, y, z = np.sin(ph) * np.cos(th), np.sin(ph) * np.sin(th), np.cos(ph)
            m = net.with_flat(x * w0 + y * w1 + z * w2)
            zv, zt = m.logits(Xva), m.logits(Xte)
            acc_va[i, j], acc_te[i, j] = topk_accuracy(zv, yva), topk_accuracy(zt, yte)
            ce_va[i, j] = cross_entropy(zv, yva)
    return acc_va, acc_te, ce_va


rows, keep = [], None
for seed in range(5):
    net = MLP.init(SIZES, "leaky_relu", use_bias=False, seed=seed)
    w0 = net.flat()
    snaps = train(net, Xtr, ytr, epochs=6, seed=seed, snapshots=[1, 6])
    w1, w2 = snaps[1].flat(), snaps[6].flat()
    acc_va, acc_te, ce_va = sphere_scan(net, w0, w1, w2)
    iv = np.unravel_index(np.argmax(acc_va), acc_va.shape)
    it = np.unravel_index(np.argmax(acc_te), acc_te.shape)
    final_te = topk_accuracy(snaps[6].logits(Xte), yte)
    rows.append({
        "seed": seed, "final_snapshot_test_acc": final_te,
        "sphere_best_on_val__test_acc": float(acc_te[iv]),
        "sphere_best_on_test__test_acc": float(acc_te[it]),
        "optimism_of_selecting_on_test": float(acc_te[it] - acc_te[iv]),
    })
    print(rows[-1])
    if seed == 0:
        keep = (acc_va, acc_te, iv, it, w0, w2, net)

summary = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != "seed"}
save_json("exp3_slices", {"grid": [NT, NP], "seeds": rows, "mean": summary,
                          "snapshots": "w0 = init, m1 = after epoch 1, m2 = after epoch 6"})
print("mean", summary)

# Figure: sphere heatmap of validation accuracy (seed 0), with both selections marked.
acc_va, acc_te, iv, it, w0, w2, net = keep
cmap = LinearSegmentedColormap.from_list("seq", SEQ)
fig, ax = plt.subplots(figsize=(8, 3.6))
im = ax.imshow(acc_va.T, origin="lower", aspect="auto", cmap=cmap, vmin=0, vmax=1,
               extent=[0, 360, 0, 180])
ax.grid(False)
ax.scatter([np.degrees(thetas[iv[0]])], [np.degrees(phis[iv[1]])], s=70, marker="o",
           facecolor="none", edgecolor=ORANGE, linewidth=2, label="best on validation")
ax.scatter([np.degrees(thetas[it[0]])], [np.degrees(phis[it[1]])], s=70, marker="x",
           color=INK2, linewidth=2, label="best on test (the old protocol)")
ax.set(xlabel="theta (degrees)", ylabel="phi (degrees)",
       title="Validation accuracy of x*w0 + y*m1 + z*m2 over the unit sphere")
fig.colorbar(im, ax=ax, label="validation accuracy")
ax.legend(loc="upper right", fontsize=8)
save_fig(fig, "exp3_sphere")

# Figure: the notebook's circle cos(s) w0 + sin(s) m2.
ss = np.linspace(0, 2 * np.pi, 361)
ce, acc = [], []
for s_ in ss:
    m = net.with_flat(np.cos(s_) * w0 + np.sin(s_) * w2)
    z = m.logits(Xva)
    ce.append(cross_entropy(z, yva))
    acc.append(topk_accuracy(z, yva))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.2), gridspec_kw={"wspace": 0.4})
a1.plot(np.degrees(ss), acc, color=BLUE)
a1.set(xlabel="s (degrees)", ylabel="validation accuracy", title="cos(s) w0 + sin(s) m2: accuracy")
a2.plot(np.degrees(ss), ce, color=AQUA)
a2.set(xlabel="s (degrees)", ylabel="validation cross-entropy", title="... and cross-entropy",
       yscale="log")
for a in (a1, a2):
    a.set_xticks([0, 90, 180, 270, 360])
save_fig(fig, "exp3_circle")
