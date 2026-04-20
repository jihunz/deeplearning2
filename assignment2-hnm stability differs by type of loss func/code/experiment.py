"""
Assignment 2 — Active Sample Theory of Hard Negative Mining
============================================================
Research Question:
  "Detection의 미검출 기반 hard example mining과 metric learning의 hard negative
   mining은 'loss > 0인 active sample에만 gradient를 흘려보낸다'는 동일 원리에서
   출발하지만, 두 setting에서 active sample의 gradient 분산과 embedding collapse
   위험은 왜 달라지는가?"

This experiment tests H1~H4:
  H1  active ratio: random 급락, hard 유지, semi-hard 중간
  H2  gradient norm ‖p-n‖: hard가 오히려 작고 variance 크다
  H3  hard mining -> embedding collapse (intra/inter 동시 0, rank 하락)
  H4  retrieval R@1: semi-hard > hard > random

NumPy-only implementation. No PyTorch needed — we implement:
  - MLP forward/backward
  - L2 normalization with Jacobian
  - Triplet loss with gradient ∂L/∂a = 2(p - n) (active) / 0 (inactive)
  - 3 mining strategies: random, semi-hard, hard
  - 7 measurements: active_ratio, |p-n|_mean, |p-n|_std,
                    intra_std, inter_dist, effective_rank, R@1, R@5

Dataset: controlled synthetic — 10-class Gaussian clusters in R^64 with
         tunable inter-class overlap.  (torch/FashionMNIST unavailable in
         sandbox; controlled synthetic is appropriate for testing the
         *structural* behavior of mining which is the paper's subject.)
"""

import os, json, time
from pathlib import Path
import numpy as np

# ------------------------------------------------------------ config
SEED_BASE  = 42
INPUT_DIM  = 64
HIDDEN1    = 128
HIDDEN2    = 64
EMB_DIM    = 16
NUM_CLASS  = 10
TRAIN_PER  = 400
TEST_PER   = 150
SIGMA      = 0.55        # class cloud std — tuned so classes overlap
MARGIN     = 0.2
LR         = 3e-3
EPOCHS     = 30
BATCH_P    = 8           # classes per batch (PK sampling)
BATCH_K    = 8           # samples per class in batch
STRATEGIES = ("random", "semi_hard", "hard")
SEEDS      = (42, 43)    # 2 seeds for mean±range; kept small for sandbox time budget

OUT   = Path(__file__).resolve().parent.parent
FIG   = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
LOG   = OUT / "code" / "results.json"
RUNLOG = OUT / "code" / "run.log"

# ------------------------------------------------------------ data
def make_data(seed):
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((NUM_CLASS, INPUT_DIM)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    def build(n_per):
        X = np.concatenate([
            centers[k] + SIGMA * rng.standard_normal((n_per, INPUT_DIM)).astype(np.float32)
            for k in range(NUM_CLASS)])
        y = np.concatenate([np.full(n_per, k, dtype=np.int64) for k in range(NUM_CLASS)])
        idx = rng.permutation(len(X))
        return X[idx], y[idx]
    Xtr, ytr = build(TRAIN_PER)
    Xte, yte = build(TEST_PER)
    return Xtr, ytr, Xte, yte

# ------------------------------------------------------------ model: MLP 64->128->64->16, L2-norm
class MLP:
    """He-init MLP with ReLU.  Last layer followed by L2 normalization.
    Forward caches activations so we can backprop the triplet-loss gradient.
    """
    def __init__(self, rng):
        def he(fan_in, fan_out):
            return rng.standard_normal((fan_in, fan_out)).astype(np.float32) * np.sqrt(2.0/fan_in)
        self.W1 = he(INPUT_DIM, HIDDEN1); self.b1 = np.zeros(HIDDEN1, dtype=np.float32)
        self.W2 = he(HIDDEN1,   HIDDEN2); self.b2 = np.zeros(HIDDEN2, dtype=np.float32)
        self.W3 = he(HIDDEN2,   EMB_DIM); self.b3 = np.zeros(EMB_DIM, dtype=np.float32)

    def params(self):
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    def forward(self, X):
        # pre-activations
        Z1 = X @ self.W1 + self.b1
        A1 = np.maximum(Z1, 0)
        Z2 = A1 @ self.W2 + self.b2
        A2 = np.maximum(Z2, 0)
        Z3 = A2 @ self.W3 + self.b3           # raw embedding, before normalization
        # L2 normalize
        norm = np.linalg.norm(Z3, axis=1, keepdims=True) + 1e-8
        E = Z3 / norm
        cache = (X, Z1, A1, Z2, A2, Z3, norm, E)
        return E, cache

    def backward(self, dE, cache):
        """Given dL/dE (B, EMB_DIM), return grads w.r.t. params and input.
        Includes Jacobian of L2 normalization:
            if e = z / ||z||,  de/dz = (I - e e^T) / ||z||.
        For backprop:  dz = (1/||z||) * (de - (de·e) e).
        """
        X, Z1, A1, Z2, A2, Z3, norm, E = cache
        # backprop through L2 normalize
        # dZ3 = (dE - (dE*E).sum(1, keepdims=True) * E) / norm
        dZ3 = (dE - (dE * E).sum(axis=1, keepdims=True) * E) / norm
        dW3 = A2.T @ dZ3
        db3 = dZ3.sum(axis=0)
        dA2 = dZ3 @ self.W3.T
        dZ2 = dA2 * (Z2 > 0)
        dW2 = A1.T @ dZ2
        db2 = dZ2.sum(axis=0)
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * (Z1 > 0)
        dW1 = X.T @ dZ1
        db1 = dZ1.sum(axis=0)
        return [dW1, db1, dW2, db2, dW3, db3]

# ------------------------------------------------------------ Adam
class Adam:
    def __init__(self, params, lr=LR, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0
    def step(self, params, grads):
        self.t += 1
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            m_hat = self.m[i] / (1 - self.b1**self.t)
            v_hat = self.v[i] / (1 - self.b2**self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

# ------------------------------------------------------------ PK batch sampler
def pk_sampler(Y, P=BATCH_P, K=BATCH_K, rng=None):
    """Yield indices of batches of size P*K with P classes × K samples each."""
    rng = rng or np.random.default_rng()
    classes = np.unique(Y)
    cls_to_idx = {c: np.where(Y == c)[0] for c in classes}
    n_batches = len(Y) // (P*K)
    for _ in range(n_batches):
        chosen = rng.choice(classes, size=P, replace=False)
        idx = []
        for c in chosen:
            pool = cls_to_idx[c]
            sel = rng.choice(pool, size=K, replace=len(pool) < K)
            idx.extend(sel.tolist())
        yield np.array(idx)

# ------------------------------------------------------------ triplet mining + loss
def pdist_sq(x):
    """For L2-normalized x, squared distance = 2 - 2 x x^T, clamped."""
    return np.maximum(2.0 - 2.0 * x @ x.T, 0.0)

def mine_triplets(E, Y, strategy, margin=MARGIN, rng=None):
    """For each anchor in batch, pick (p, n) and return arrays:
       a_idx, p_idx, n_idx (all length = #triplets used in this step, at most B).
       Active-only: we still emit inactive triplets; the caller uses the mask.
    Positive: random sample-same-class (per Step-3 design).
    Negative: random / hard / semi-hard.
    """
    rng = rng or np.random.default_rng()
    B = E.shape[0]
    D = pdist_sq(E)
    same = Y[:, None] == Y[None, :]
    eye = np.eye(B, dtype=bool)
    pos_mask = same & ~eye
    neg_mask = ~same
    a_list, p_list, n_list = [], [], []
    for a in range(B):
        pi = np.where(pos_mask[a])[0]
        ni = np.where(neg_mask[a])[0]
        if len(pi) == 0 or len(ni) == 0:
            continue
        p = rng.choice(pi)                        # random positive
        dap = D[a, p]
        if strategy == "random":
            n = rng.choice(ni)
        elif strategy == "hard":
            n = ni[np.argmin(D[a, ni])]
        elif strategy == "semi_hard":
            dan = D[a, ni]
            cand = ni[(dan > dap) & (dan < dap + margin)]
            if len(cand) == 0:
                # fallback: closest negative that is still further than positive
                further = ni[dan > dap]
                n = further[np.argmin(D[a, further])] if len(further) else ni[np.argmax(D[a, ni])]
            else:
                n = rng.choice(cand)
        else:
            raise ValueError(strategy)
        a_list.append(a); p_list.append(p); n_list.append(n)
    return np.array(a_list), np.array(p_list), np.array(n_list)

def triplet_forward_backward(E, a_idx, p_idx, n_idx, margin=MARGIN):
    """Compute triplet loss, gradient w.r.t. E, and observables.

    L_i = max(0, d(a,p) - d(a,n) + m)
    For L2-normalized E with squared L2 distance: d(x,y) = 2 - 2 x·y.
      ∂L/∂e_a = -2(p - n) * [active]  (anchor pulled toward p, pushed from n)
      ∂L/∂e_p = -2(a)          * [active] ... but we combine:
    Using d(x,y) = ||x - y||^2 under L2-norm (effectively same as 2-2xy up to
    constants on the sphere):
      ∂L/∂a = 2(a-p) - 2(a-n) = 2(n - p)        ... (active)
      ∂L/∂p = 2(p - a)                          ... (active)
      ∂L/∂n = 2(a - n)                          ... (active)
    So per-triplet gradient-proxy magnitude used in the paper's theory is:
         ||p - n||    (anchor-side gradient norm, divided by 2)
    """
    Ea = E[a_idx]; Ep = E[p_idx]; En = E[n_idx]
    dap = np.sum((Ea - Ep) ** 2, axis=1)
    dan = np.sum((Ea - En) ** 2, axis=1)
    raw = dap - dan + margin
    active = raw > 0                     # hinge mask
    loss_per = np.maximum(raw, 0.0)
    # observables
    pn_norm = np.linalg.norm(Ep - En, axis=1)          # ||p - n||  per triplet
    active_mask = active
    # gradient on E (batch-sized accumulator)
    dE = np.zeros_like(E)
    if active.any():
        amask = active.astype(np.float32)[:, None]
        np.add.at(dE, a_idx, amask * 2.0 * (En - Ep))
        np.add.at(dE, p_idx, amask * 2.0 * (Ep - Ea))
        np.add.at(dE, n_idx, amask * 2.0 * (Ea - En))
    # mean loss (over non-zero? over all? use over-all to preserve signal magnitude)
    loss_mean = loss_per.mean() if len(loss_per) else 0.0
    dE /= max(len(a_idx), 1)
    return loss_mean, dE, active_mask, pn_norm

# ------------------------------------------------------------ metrics
def eval_retrieval(model, Xq, Yq, Xg, Yg, ks=(1, 5)):
    Eq, _ = model.forward(Xq)
    Eg, _ = model.forward(Xg)
    D = 2.0 - 2.0 * Eq @ Eg.T                         # squared cos-dist (monotone with L2^2)
    # nearest-k
    idx = np.argpartition(D, ks[-1], axis=1)[:, :ks[-1]]
    # sort just the topk by true value for R@1 accuracy
    order = np.argsort(np.take_along_axis(D, idx, axis=1), axis=1)
    idx = np.take_along_axis(idx, order, axis=1)
    res = {}
    for k in ks:
        hit = (Yg[idx[:, :k]] == Yq[:, None]).any(axis=1).mean()
        res[f"R@{k}"] = float(hit)
    return res

def collapse_stats(E, Y):
    """Return (intra_std, inter_dist, effective_rank).
       intra_std     = mean over classes of per-class std across dims, averaged
       inter_dist    = mean pairwise L2 distance between class centroids
       effective_rank= exp(entropy of normalized singular value spectrum)
    """
    classes = np.unique(Y)
    centroids = np.stack([E[Y == c].mean(axis=0) for c in classes])
    intra = np.mean([E[Y == c].std(axis=0).mean() for c in classes])
    # pairwise centroid distance
    diffs = centroids[:, None, :] - centroids[None, :, :]
    D = np.sqrt((diffs ** 2).sum(axis=-1))
    triu = D[np.triu_indices_from(D, k=1)]
    inter = float(triu.mean())
    # effective rank
    s = np.linalg.svd(E - E.mean(axis=0, keepdims=True), compute_uv=False)
    s2 = s ** 2
    p = s2 / (s2.sum() + 1e-12)
    H = -(p * np.log(p + 1e-12)).sum()
    eff_rank = float(np.exp(H))
    return float(intra), inter, eff_rank

# ------------------------------------------------------------ training loop
def run_one(strategy, seed, Xtr, Ytr, Xte, Yte, logf):
    rng = np.random.default_rng(seed)
    model = MLP(rng)
    opt = Adam(model.params())
    hist = {"loss": [], "active_ratio": [], "pn_mean": [], "pn_std": [],
            "intra": [], "inter": [], "eff_rank": [], "R@1": [], "R@5": []}
    for ep in range(1, EPOCHS + 1):
        loss_s = active_s = pn_m_s = pn_s_s = n_batch = 0
        for idx in pk_sampler(Ytr, P=BATCH_P, K=BATCH_K, rng=rng):
            Xb = Xtr[idx]; Yb = Ytr[idx]
            E, cache = model.forward(Xb)
            a, p, n = mine_triplets(E, Yb, strategy, rng=rng)
            if len(a) == 0:
                continue
            loss, dE, act_mask, pn_norm = triplet_forward_backward(E, a, p, n)
            grads = model.backward(dE, cache)
            opt.step(model.params(), grads)
            # gradient clipping to stabilize hard
            # (logging first)
            loss_s   += loss
            active_s += act_mask.mean()
            if act_mask.any():
                pn_m_s += float(pn_norm[act_mask].mean())
                pn_s_s += float(pn_norm[act_mask].std())
            n_batch += 1
        # eval
        Ete, _ = model.forward(Xte)
        intra, inter, eff = collapse_stats(Ete, Yte)
        r = eval_retrieval(model, Xte, Yte, Xtr, Ytr)
        hist["loss"].append(loss_s / max(n_batch, 1))
        hist["active_ratio"].append(active_s / max(n_batch, 1))
        hist["pn_mean"].append(pn_m_s / max(n_batch, 1))
        hist["pn_std"].append(pn_s_s / max(n_batch, 1))
        hist["intra"].append(intra); hist["inter"].append(inter); hist["eff_rank"].append(eff)
        hist["R@1"].append(r["R@1"]); hist["R@5"].append(r["R@5"])
        msg = (f"[{strategy:9s} seed{seed}] ep{ep:02d}  loss={hist['loss'][-1]:.4f}  "
               f"act={hist['active_ratio'][-1]:.3f}  |p-n|={hist['pn_mean'][-1]:.3f}"
               f"±{hist['pn_std'][-1]:.3f}  intra={intra:.3f}  inter={inter:.3f}  "
               f"rank={eff:.2f}  R@1={r['R@1']:.3f}")
        print(msg); logf.write(msg + "\n"); logf.flush()
    # final embedding for PCA viz
    Ete, _ = model.forward(Xte)
    return hist, Ete

def main():
    t0 = time.time()
    results = {s: {} for s in STRATEGIES}
    final_emb = {}
    with open(RUNLOG, "w") as logf:
        # data shared across strategies but per-seed so each run sees same data
        for seed in SEEDS:
            Xtr, Ytr, Xte, Yte = make_data(seed)
            for s in STRATEGIES:
                print(f"\n===== strategy={s} seed={seed} =====")
                logf.write(f"\n===== strategy={s} seed={seed} =====\n")
                hist, Ete = run_one(s, seed, Xtr, Ytr, Xte, Yte, logf)
                results[s][str(seed)] = hist
                if seed == SEEDS[0]:
                    final_emb[s] = dict(E=Ete.tolist(), Y=Yte.tolist())
    meta = dict(
        input_dim=INPUT_DIM, hidden=(HIDDEN1, HIDDEN2), emb_dim=EMB_DIM,
        num_class=NUM_CLASS, train_per=TRAIN_PER, test_per=TEST_PER, sigma=SIGMA,
        margin=MARGIN, lr=LR, epochs=EPOCHS, P=BATCH_P, K=BATCH_K,
        seeds=list(SEEDS), strategies=list(STRATEGIES), elapsed=round(time.time()-t0,1),
    )
    results["_meta"] = meta
    results["_final_emb"] = final_emb
    LOG.write_text(json.dumps(results, indent=2))
    print("Saved:", LOG, "  elapsed:", meta["elapsed"], "s")

if __name__ == "__main__":
    main()
