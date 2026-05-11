"""
A_03 — BALD vs Softmax Entropy as Acquisition Functions under Domain Shift
Author: Jihun Jang  (jhjang@vetec.co.kr)

Goal
----
Empirically test the claim that softmax predictive entropy (H_pred), the
acquisition signal used in the user's KIISE/AICompS active-domain-adaptation
pipeline, conflates aleatoric and epistemic uncertainty, and that
MC-Dropout BALD recovers the epistemic component which is the signal
active sample selection should optimize under domain shift.

We deliberately use NumPy only (no PyTorch) so the math is transparent
and the run finishes in <2 minutes on a single CPU.

We design two cross-domain regimes:
    REGIME A — covariate shift only:
        Source = N(±1, 0)+N(0,σ),  Target = N(±1, Δ)+N(0,σ).
        The optimal decision boundary x_1=0 is preserved.
    REGIME B — concept + covariate shift:
        Target classes are *rotated* by 90° relative to source so the
        true boundary in target is x_2 = 0. The source-trained model is
        catastrophically wrong on target — this is the regime where
        epistemic uncertainty dominates.

For each regime we sweep Δ and compare three acquisitions in a real
active learning loop: Random, H_pred (entropy), BALD. After selection,
the model is retrained on Source∪{selected} and target-test accuracy is
recorded.

Outputs: results.json, figures/fig{1..5}.pdf
"""
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SEED = 42

# ===========================================================================
# Data generation
# ===========================================================================
def make_blobs(n, centers, sigma, rng):
    blobs = []
    labels = []
    for c, mu in enumerate(centers):
        x = rng.normal(loc=mu, scale=sigma, size=(n, 2))
        blobs.append(x)
        labels.append(np.full(n, c))
    X = np.vstack(blobs).astype(np.float32)
    y = np.concatenate(labels).astype(np.int64)
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


def source_data(n_per_class, sigma, rng):
    return make_blobs(n_per_class, [(-1.0, 0.0), (1.0, 0.0)], sigma, rng)


def target_data_A(n_per_class, sigma, delta, rng):
    """Covariate shift: vertical shift by delta."""
    return make_blobs(n_per_class, [(-1.0, delta), (1.0, delta)], sigma, rng)


def target_data_B(n_per_class, sigma, delta, rng):
    """Concept shift: rotate class layout by 90 degrees, also translate."""
    # Now class 0 is below, class 1 is above; boundary becomes x_2 = delta
    return make_blobs(n_per_class, [(0.0, delta - 1.0), (0.0, delta + 1.0)], sigma, rng)


# ===========================================================================
# NumPy MLP with Dropout
# ===========================================================================
def relu(x):
    return np.maximum(0.0, x)

def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    ez = np.exp(z)
    return ez / ez.sum(axis=-1, keepdims=True)

def log_softmax(z):
    """LogSumExp-stable log softmax (lecture pp. 198-229)."""
    m = z.max(axis=-1, keepdims=True)
    return z - m - np.log(np.exp(z - m).sum(axis=-1, keepdims=True))


class MLP:
    def __init__(self, hidden=64, p_drop=0.3, rng=None):
        self.p = p_drop
        self.hidden = hidden
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self._init()
        self._dropout_on = False

    def _init(self):
        h = self.hidden
        s1 = np.sqrt(2.0 / 2)
        s2 = np.sqrt(2.0 / h)
        s3 = np.sqrt(2.0 / h)
        self.W1 = self.rng.normal(0, s1, (2, h)).astype(np.float32)
        self.b1 = np.zeros(h, dtype=np.float32)
        self.W2 = self.rng.normal(0, s2, (h, h)).astype(np.float32)
        self.b2 = np.zeros(h, dtype=np.float32)
        self.W3 = self.rng.normal(0, s3, (h, 2)).astype(np.float32)
        self.b3 = np.zeros(2, dtype=np.float32)

    def _dropout(self, a, training):
        if training or self._dropout_on:
            mask = (self.rng.uniform(size=a.shape) > self.p).astype(np.float32) / (1 - self.p)
            return a * mask, mask
        return a, np.ones_like(a)

    def forward(self, X, training=False):
        z1 = X @ self.W1 + self.b1
        a1 = relu(z1)
        a1, m1 = self._dropout(a1, training)
        z2 = a1 @ self.W2 + self.b2
        a2 = relu(z2)
        a2, m2 = self._dropout(a2, training)
        logits = a2 @ self.W3 + self.b3
        cache = (X, z1, a1, m1, z2, a2, m2)
        return logits, cache

    def loss_and_grad(self, X, y, lam=1e-4):
        N = X.shape[0]
        logits, cache = self.forward(X, training=True)
        log_p = log_softmax(logits)
        nll = -log_p[np.arange(N), y].mean()
        l2 = lam * (np.square(self.W1).sum() + np.square(self.W2).sum() + np.square(self.W3).sum())
        loss = nll + l2

        X_, z1, a1, m1, z2, a2, m2 = cache
        p = np.exp(log_p)
        dlogits = p
        dlogits[np.arange(N), y] -= 1
        dlogits /= N

        dW3 = a2.T @ dlogits + 2 * lam * self.W3
        db3 = dlogits.sum(0)
        da2 = dlogits @ self.W3.T
        da2 = da2 * m2 * (z2 > 0)
        dW2 = a1.T @ da2 + 2 * lam * self.W2
        db2 = da2.sum(0)
        da1 = da2 @ self.W2.T
        da1 = da1 * m1 * (z1 > 0)
        dW1 = X_.T @ da1 + 2 * lam * self.W1
        db1 = da1.sum(0)
        return loss, (dW1, db1, dW2, db2, dW3, db3)

    def predict_logits(self, X, mc_samples=1, dropout_on=False):
        self._dropout_on = dropout_on
        outs = []
        for _ in range(mc_samples):
            logits, _ = self.forward(X, training=False)
            outs.append(logits)
        self._dropout_on = False
        return np.stack(outs, axis=0)

    def state(self):
        return tuple(p.copy() for p in [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3])

    def load(self, st):
        self.W1, self.b1, self.W2, self.b2, self.W3, self.b3 = (p.copy() for p in st)


def train(model, X, y, Xv, yv, epochs=200, lr=0.05, batch=64, patience=30, verbose=False):
    best_val, best_state, plateau = np.inf, None, 0
    history = {"train": [], "val": []}
    N = X.shape[0]
    for ep in range(epochs):
        perm = model.rng.permutation(N)
        Xs, ys = X[perm], y[perm]
        losses = []
        for i in range(0, N, batch):
            xb, yb = Xs[i:i+batch], ys[i:i+batch]
            loss, grads = model.loss_and_grad(xb, yb)
            dW1, db1, dW2, db2, dW3, db3 = grads
            model.W1 -= lr * dW1; model.b1 -= lr * db1
            model.W2 -= lr * dW2; model.b2 -= lr * db2
            model.W3 -= lr * dW3; model.b3 -= lr * db3
            losses.append(loss)
        val_logits, _ = model.forward(Xv, training=False)
        vl = -log_softmax(val_logits)[np.arange(len(yv)), yv].mean()
        history["train"].append(float(np.mean(losses)))
        history["val"].append(float(vl))
        if vl < best_val - 1e-4:
            best_val, best_state, plateau = vl, model.state(), 0
        else:
            plateau += 1
            if plateau > patience:
                break
    if best_state is not None:
        model.load(best_state)
    return history


# ===========================================================================
# Acquisition functions
# ===========================================================================
def entropy_from_probs(p, eps=1e-12):
    return -np.sum(p * np.log(p + eps), axis=-1)


def mc_acquisitions(model, X, n_mc=30):
    logits = model.predict_logits(X, mc_samples=n_mc, dropout_on=True)  # (T,N,C)
    probs = np.exp(log_softmax(logits))                                  # (T,N,C)
    p_bar = probs.mean(axis=0)
    H_pred = entropy_from_probs(p_bar)
    H_alea = entropy_from_probs(probs).mean(axis=0)
    BALD = H_pred - H_alea
    return {"H_pred": H_pred, "H_alea": H_alea, "BALD": BALD, "p_bar": p_bar}


# ===========================================================================
# Active learning loop
# ===========================================================================
def select_topk(scores, k):
    return np.argsort(-scores)[:k]


def select_random(n, k, rng):
    return rng.choice(n, size=k, replace=False)


def evaluate(model, X, y):
    logits, _ = model.forward(X, training=False)
    pred = logits.argmax(-1)
    return float((pred == y).mean())


def train_from_scratch(Xtr, ytr, Xv, yv, seed=0, hidden=32, p_drop=0.3,
                       epochs=80, lr=0.08, batch=64, patience=15):
    rng = np.random.default_rng(seed)
    m = MLP(hidden=hidden, p_drop=p_drop, rng=rng)
    train(m, Xtr, ytr, Xv, yv, epochs=epochs, lr=lr, batch=batch, patience=patience)
    return m


# ===========================================================================
# Experiments
# ===========================================================================
def regime_label(reg):
    return {"A": "Covariate shift", "B": "Concept shift"}[reg]


def run_one(regime, delta, K_list=(0, 10, 30, 60, 120), seeds=(0, 1, 2),
            n_pool=400, n_test=400, n_src=400, sigma=0.7, n_mc=30):
    """Return dict[K][acquisition][seed] = target_test_acc."""
    out = {K: {a: [] for a in ["random", "H_pred", "BALD"]} for K in K_list}
    info = {}
    for s in seeds:
        rng = np.random.default_rng(SEED + s)
        Xs, ys = source_data(n_src, sigma, rng)
        Xs_va, ys_va = source_data(n_src // 4, sigma, rng)
        if regime == "A":
            Xt_pool, yt_pool = target_data_A(n_pool, sigma, delta, rng)
            Xt_te, yt_te = target_data_A(n_test, sigma, delta, rng)
        else:
            Xt_pool, yt_pool = target_data_B(n_pool, sigma, delta, rng)
            Xt_te, yt_te = target_data_B(n_test, sigma, delta, rng)

        # 1. Train base model on source
        base = train_from_scratch(Xs, ys, Xs_va, ys_va, seed=SEED + s)
        zero_acc = evaluate(base, Xt_te, yt_te)
        # 2. Compute acquisitions ONCE on the pool (using base model)
        acq = mc_acquisitions(base, Xt_pool, n_mc=n_mc)
        # 3. For each K and acquisition, build augmented training set and retrain
        for K in K_list:
            for name in ["random", "H_pred", "BALD"]:
                if K == 0:
                    out[K][name].append(zero_acc)
                    continue
                if name == "random":
                    idx = select_random(len(Xt_pool), K, rng)
                else:
                    idx = select_topk(acq[name], K)
                Xtr = np.vstack([Xs, Xt_pool[idx]])
                ytr = np.concatenate([ys, yt_pool[idx]])
                m = train_from_scratch(Xtr, ytr, Xs_va, ys_va, seed=SEED + s + 1000)
                acc = evaluate(m, Xt_te, yt_te)
                out[K][name].append(acc)
        info["base_target_acc_seed%d" % s] = zero_acc
    return out, info


def aggregate(curve):
    """curve: dict[K][name] = list of accs. Return means and stds."""
    K_list = sorted(curve.keys())
    res = {}
    for name in curve[K_list[0]].keys():
        means = [float(np.mean(curve[K][name])) for K in K_list]
        stds = [float(np.std(curve[K][name])) for K in K_list]
        res[name] = {"K": K_list, "mean": means, "std": stds}
    return res


# ===========================================================================
# Diagnostics: acquisition heatmaps + entropy/BALD decomposition
# ===========================================================================
def diagnostics(out_dir, regime="B", delta=2.0, sigma=0.7, n_mc=30):
    fig_dir = os.path.join(out_dir, "..", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    rng = np.random.default_rng(SEED + 100)
    Xs, ys = source_data(400, sigma, rng)
    Xs_va, ys_va = source_data(100, sigma, rng)
    if regime == "A":
        Xt_pool, yt_pool = target_data_A(300, sigma, delta, rng)
    else:
        Xt_pool, yt_pool = target_data_B(300, sigma, delta, rng)
    base = train_from_scratch(Xs, ys, Xs_va, ys_va, seed=SEED + 100)
    src_acc = evaluate(base, Xs_va, ys_va)
    tgt_acc = evaluate(base, Xt_pool, yt_pool)

    # Heatmaps over a 2D grid
    grid_x = np.linspace(-3, 3, 90)
    grid_y = np.linspace(-2, 5, 90)
    XX, YY = np.meshgrid(grid_x, grid_y)
    grid = np.stack([XX.ravel(), YY.ravel()], axis=1).astype(np.float32)
    a = mc_acquisitions(base, grid, n_mc=n_mc)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0), sharey=True)
    titles = [
        r"$H_{\mathrm{pred}}$ (predictive entropy)",
        r"$H_{\mathrm{alea}} = \mathbb{E}_\theta H[p(y|x,\theta)]$",
        r"BALD $= H_{\mathrm{pred}} - H_{\mathrm{alea}}$ (epistemic)",
    ]
    for ax, key, t in zip(axes, ["H_pred", "H_alea", "BALD"], titles):
        Z = a[key].reshape(XX.shape)
        im = ax.pcolormesh(XX, YY, Z, cmap="viridis", shading="auto")
        ax.contour(XX, YY, Z, levels=8, colors="white", linewidths=0.4, alpha=0.6)
        ax.scatter(Xs[ys == 0, 0], Xs[ys == 0, 1], s=4, c="white", alpha=0.4, marker="o", label="src 0")
        ax.scatter(Xs[ys == 1, 0], Xs[ys == 1, 1], s=4, c="white", alpha=0.4, marker="x", label="src 1")
        ax.scatter(Xt_pool[yt_pool == 0, 0], Xt_pool[yt_pool == 0, 1], s=8, c="cyan", alpha=0.6, marker="^", label="tgt 0")
        ax.scatter(Xt_pool[yt_pool == 1, 0], Xt_pool[yt_pool == 1, 1], s=8, c="orange", alpha=0.6, marker="v", label="tgt 1")
        ax.set_xlabel(r"$x_1$"); ax.set_title(t)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    axes[0].set_ylabel(r"$x_2$")
    axes[0].legend(fontsize=7, loc="upper left", framealpha=0.7)
    fig.suptitle(f"{regime_label(regime)} (Δ={delta}). Src acc={src_acc:.2f}, Tgt acc={tgt_acc:.2f}",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"fig2_acquisition_maps_{regime}.pdf"))
    plt.close(fig)


# ===========================================================================
# Main driver
# ===========================================================================
def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    fig_dir = os.path.join(out_dir, "..", "figures")
    os.makedirs(fig_dir, exist_ok=True)

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    t0 = time.time()
    results = {"seed": SEED}

    # ----- Distribution illustration -----
    rng = np.random.default_rng(SEED)
    Xs, ys = source_data(300, 0.7, rng)
    XtA, ytA = target_data_A(300, 0.7, 2.5, rng)
    XtB, ytB = target_data_B(300, 0.7, 2.5, rng)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, Xt, yt, title in zip(axes, [XtA, XtB], [ytA, ytB],
                                 ["Regime A — covariate shift (Δ=2.5)",
                                  "Regime B — concept shift (Δ=2.5)"]):
        ax.scatter(Xs[ys == 0, 0], Xs[ys == 0, 1], s=8, alpha=0.5, c="#2E86AB", label="src class 0")
        ax.scatter(Xs[ys == 1, 0], Xs[ys == 1, 1], s=8, alpha=0.5, c="#E63946", label="src class 1")
        ax.scatter(Xt[yt == 0, 0], Xt[yt == 0, 1], s=14, alpha=0.7, marker="^", c="#1B4965", edgecolor="black", linewidth=0.3, label="tgt class 0")
        ax.scatter(Xt[yt == 1, 0], Xt[yt == 1, 1], s=14, alpha=0.7, marker="v", c="#9D0208", edgecolor="black", linewidth=0.3, label="tgt class 1")
        ax.axvline(0, color="k", linestyle="--", linewidth=0.7, alpha=0.5)
        ax.set_xlabel(r"$x_1$"); ax.set_ylabel(r"$x_2$"); ax.set_title(title)
        ax.legend(fontsize=7, loc="upper left", framealpha=0.7)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig1_distributions.pdf"))
    plt.close(fig)

    # ----- Diagnostics for both regimes -----
    diagnostics(out_dir, regime="A", delta=2.5)
    diagnostics(out_dir, regime="B", delta=2.5)

    # ----- Active learning curves -----
    K_list = (0, 20, 50, 100)
    deltas_A = [1.5, 3.0]
    deltas_B = [1.5, 3.0]
    al_results = {}
    for reg, deltas in [("A", deltas_A), ("B", deltas_B)]:
        for d in deltas:
            label = f"{reg}_d{d}"
            print(f"Running regime {reg}, delta={d} ...", flush=True)
            curve, info = run_one(regime=reg, delta=d, K_list=K_list, seeds=(0, 1))
            al_results[label] = aggregate(curve)
            al_results[label]["info"] = info
    results["active_learning"] = al_results
    # checkpoint partial results
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # ----- Plot AL curves -----
    fig, axes = plt.subplots(2, len(deltas_A), figsize=(5*len(deltas_A), 7), sharey=True, squeeze=False)
    for row, reg, deltas in [(0, "A", deltas_A), (1, "B", deltas_B)]:
        for col, d in enumerate(deltas):
            ax = axes[row, col]
            label = f"{reg}_d{d}"
            agg = al_results[label]
            for name, color, marker in [("random", "#888888", "o"),
                                         ("H_pred", "#2E86AB", "s"),
                                         ("BALD",  "#E63946", "^")]:
                m = np.array(agg[name]["mean"])
                s = np.array(agg[name]["std"])
                ax.plot(agg[name]["K"], m, color=color, marker=marker, label=name)
                ax.fill_between(agg[name]["K"], m - s, m + s, color=color, alpha=0.15)
            ax.set_title(f"Regime {reg} ({regime_label(reg)}), Δ={d}")
            ax.set_xlabel("Labelling budget K")
            if col == 0:
                ax.set_ylabel("Target test accuracy")
            ax.grid(True, alpha=0.3)
            if row == 0 and col == 0:
                ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig3_active_learning_curves.pdf"))
    plt.close(fig)

    # ----- Δ-sweep at fixed K=50: BALD - H_pred gap -----
    delta_sweep = [0.0, 1.0, 2.0, 3.0]
    K_fix = 50
    sweep_res = {"A": {"random": [], "H_pred": [], "BALD": []},
                 "B": {"random": [], "H_pred": [], "BALD": []}}
    for reg in ["A", "B"]:
        for d in delta_sweep:
            print(f"Δ-sweep regime {reg}, delta={d}", flush=True)
            curve, _ = run_one(regime=reg, delta=d, K_list=(K_fix,), seeds=(0, 1))
            for name in ["random", "H_pred", "BALD"]:
                sweep_res[reg][name].append((float(np.mean(curve[K_fix][name])),
                                             float(np.std(curve[K_fix][name]))))
    results["delta_sweep_K30"] = {reg: {name: [{"delta": d, "mean": v[0], "std": v[1]}
                                                 for d, v in zip(delta_sweep, vs)]
                                          for name, vs in sweep_res[reg].items()}
                                    for reg in ["A", "B"]}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, reg in zip(axes, ["A", "B"]):
        for name, color, marker in [("random", "#888888", "o"),
                                     ("H_pred", "#2E86AB", "s"),
                                     ("BALD",  "#E63946", "^")]:
            ms = np.array([v[0] for v in sweep_res[reg][name]])
            ss = np.array([v[1] for v in sweep_res[reg][name]])
            ax.plot(delta_sweep, ms, color=color, marker=marker, label=name)
            ax.fill_between(delta_sweep, ms - ss, ms + ss, color=color, alpha=0.15)
        ax.set_xlabel("Domain shift strength Δ")
        ax.set_title(f"Regime {reg} — {regime_label(reg)}, K={K_fix}")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Target test accuracy")
    axes[0].legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig4_delta_sweep.pdf"))
    plt.close(fig)

    # ----- Save -----
    results["walltime_sec"] = time.time() - t0
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps({k: v for k, v in results.items() if k not in {"active_learning", "delta_sweep_K30"}}, indent=2))
    print("Wall time: %.1f s" % results["walltime_sec"])


def stage1_distributions_and_diagnostics(out_dir):
    fig_dir = os.path.join(out_dir, "..", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    rng = np.random.default_rng(SEED)
    Xs, ys = source_data(300, 0.7, rng)
    XtA, ytA = target_data_A(300, 0.7, 2.5, rng)
    XtB, ytB = target_data_B(300, 0.7, 2.5, rng)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, Xt, yt, title in zip(axes, [XtA, XtB], [ytA, ytB],
                                 ["Regime A — covariate shift (Δ=2.5)",
                                  "Regime B — concept shift (Δ=2.5)"]):
        ax.scatter(Xs[ys == 0, 0], Xs[ys == 0, 1], s=8, alpha=0.5, c="#2E86AB", label="src class 0")
        ax.scatter(Xs[ys == 1, 0], Xs[ys == 1, 1], s=8, alpha=0.5, c="#E63946", label="src class 1")
        ax.scatter(Xt[yt == 0, 0], Xt[yt == 0, 1], s=14, alpha=0.7, marker="^", c="#1B4965", edgecolor="black", linewidth=0.3, label="tgt class 0")
        ax.scatter(Xt[yt == 1, 0], Xt[yt == 1, 1], s=14, alpha=0.7, marker="v", c="#9D0208", edgecolor="black", linewidth=0.3, label="tgt class 1")
        ax.axvline(0, color="k", linestyle="--", linewidth=0.7, alpha=0.5)
        ax.set_xlabel(r"$x_1$"); ax.set_ylabel(r"$x_2$"); ax.set_title(title)
        ax.legend(fontsize=7, loc="upper left", framealpha=0.7)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig1_distributions.pdf"))
    plt.close(fig)
    diagnostics(out_dir, regime="A", delta=2.5)
    diagnostics(out_dir, regime="B", delta=2.5)


def stage2_active_learning_and_save(out_dir):
    fig_dir = os.path.join(out_dir, "..", "figures")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    K_list = (0, 20, 50, 100)
    deltas_A = [1.5, 3.0]
    deltas_B = [1.5, 3.0]
    al_results = {}
    for reg, deltas in [("A", deltas_A), ("B", deltas_B)]:
        for d in deltas:
            label = f"{reg}_d{d}"
            print(f"AL regime {reg}, delta={d} ...", flush=True)
            curve, info = run_one(regime=reg, delta=d, K_list=K_list, seeds=(0, 1))
            al_results[label] = aggregate(curve)
            al_results[label]["info"] = info
    with open(os.path.join(out_dir, "results_al.json"), "w") as f:
        json.dump(al_results, f, indent=2)

    fig, axes = plt.subplots(2, len(deltas_A), figsize=(5*len(deltas_A), 7), sharey=True, squeeze=False)
    for row, reg, deltas in [(0, "A", deltas_A), (1, "B", deltas_B)]:
        for col, d in enumerate(deltas):
            ax = axes[row, col]
            label = f"{reg}_d{d}"
            agg = al_results[label]
            for name, color, marker in [("random", "#888888", "o"),
                                         ("H_pred", "#2E86AB", "s"),
                                         ("BALD",  "#E63946", "^")]:
                m = np.array(agg[name]["mean"])
                s = np.array(agg[name]["std"])
                ax.plot(agg[name]["K"], m, color=color, marker=marker, label=name)
                ax.fill_between(agg[name]["K"], m - s, m + s, color=color, alpha=0.15)
            ax.set_title(f"Regime {reg} ({regime_label(reg)}), Δ={d}")
            ax.set_xlabel("Labelling budget K")
            if col == 0:
                ax.set_ylabel("Target test accuracy")
            ax.grid(True, alpha=0.3)
            if row == 0 and col == 0:
                ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig3_active_learning_curves.pdf"))
    plt.close(fig)


def stage3_delta_sweep_and_save(out_dir):
    fig_dir = os.path.join(out_dir, "..", "figures")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    delta_sweep = [0.0, 1.0, 2.0, 3.0]
    K_fix = 50
    sweep_res = {"A": {"random": [], "H_pred": [], "BALD": []},
                 "B": {"random": [], "H_pred": [], "BALD": []}}
    for reg in ["A", "B"]:
        for d in delta_sweep:
            print(f"Δ-sweep regime {reg}, delta={d}", flush=True)
            curve, _ = run_one(regime=reg, delta=d, K_list=(K_fix,), seeds=(0, 1))
            for name in ["random", "H_pred", "BALD"]:
                sweep_res[reg][name].append((float(np.mean(curve[K_fix][name])),
                                             float(np.std(curve[K_fix][name]))))
    with open(os.path.join(out_dir, "results_sweep.json"), "w") as f:
        json.dump({reg: {n: [{"delta": d, "mean": v[0], "std": v[1]} for d, v in zip(delta_sweep, vs)]
                          for n, vs in sweep_res[reg].items()} for reg in ["A", "B"]}, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, reg in zip(axes, ["A", "B"]):
        for name, color, marker in [("random", "#888888", "o"),
                                     ("H_pred", "#2E86AB", "s"),
                                     ("BALD",  "#E63946", "^")]:
            ms = np.array([v[0] for v in sweep_res[reg][name]])
            ss = np.array([v[1] for v in sweep_res[reg][name]])
            ax.plot(delta_sweep, ms, color=color, marker=marker, label=name)
            ax.fill_between(delta_sweep, ms - ss, ms + ss, color=color, alpha=0.15)
        ax.set_xlabel("Domain shift strength Δ")
        ax.set_title(f"Regime {reg} — {regime_label(reg)}, K={K_fix}")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Target test accuracy")
    axes[0].legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig4_delta_sweep.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    import sys
    out_dir = os.path.dirname(os.path.abspath(__file__))
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    t0 = time.time()
    if stage in ("1", "all"):
        stage1_distributions_and_diagnostics(out_dir)
        print(f"stage1 done in {time.time()-t0:.1f}s")
    if stage in ("2", "all"):
        t = time.time()
        stage2_active_learning_and_save(out_dir)
        print(f"stage2 done in {time.time()-t:.1f}s")
    if stage in ("3", "all"):
        t = time.time()
        stage3_delta_sweep_and_save(out_dir)
        print(f"stage3 done in {time.time()-t:.1f}s")
