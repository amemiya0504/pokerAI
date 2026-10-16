"""Training, evaluation and checkpoint/resume for the policy network.

Loss: reach-weighted cross-entropy against the solver's mixed strategy
(equivalent to KL up to the targets' entropy). The epoch with the lowest
validation KL is kept; the reported metrics come from the separate test
spots, which never influence training or selection. Metrics:

* KL(solver || model), reach-weighted, in nats;
* EV regret vs the solver: sum_a (solver_a - model_a) * EV_a, using the
  solver's per-combo action EVs, as % of the pot at the start of the spot;
* the same metrics for two baselines (uniform over legal actions, and the
  average solver strategy for the same street and node).

Determinism: fixed seeds, deterministic algorithms and a per-epoch shuffle
generator derived from (seed, epoch), so resuming from a checkpoint gives the
same weights as an uninterrupted run (on the same device).
"""

from __future__ import annotations

import json
import os
import platform
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402

from .data import Tensors  # noqa: E402
from .model import PolicyNet  # noqa: E402

TARGETS = {"kl_nats": 0.15, "ev_regret_pct_pot": 1.0}  # set before training (README)


@dataclass(frozen=True)
class TrainConfig:
    features: str
    epochs: int = 12
    batch_size: int = 8192
    lr: float = 2e-3
    weight_decay: float = 1e-4
    hidden: int = 256
    layers: int = 3
    range_dim: int = 64
    dropout: float = 0.0
    seed: int = 0


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def weighted_metrics(logp: torch.Tensor, target: torch.Tensor, ev: torch.Tensor,
                     evmask: torch.Tensor, weight: torch.Tensor, pot: torch.Tensor) -> dict[str, float]:
    """Sums (not means) so batches can be accumulated: kl, regret, and weights."""
    p = logp.exp()
    t_log = torch.where(target > 0, target.clamp_min(1e-12).log(), torch.zeros_like(target))
    finite_logp = torch.where(target > 0, logp, torch.zeros_like(logp))
    kl = (target * (t_log - finite_logp)).sum(1)
    regret = ((target - torch.nan_to_num(p)) * ev).sum(1) / pot * 100
    w_ev = weight * evmask
    return {"kl": float((weight * kl).sum()), "w": float(weight.sum()),
            "regret": float((w_ev * regret).sum()), "w_ev": float(w_ev.sum())}


def _finish(acc: dict[str, float]) -> dict[str, float]:
    return {"kl_nats": acc["kl"] / acc["w"], "ev_regret_pct_pot": acc["regret"] / acc["w_ev"]}


def _accumulate(acc: dict[str, float] | None, m: dict[str, float]) -> dict[str, float]:
    return dict(m) if acc is None else {k: acc[k] + m[k] for k in acc}


@torch.no_grad()
def evaluate(model: PolicyNet, data: Tensors, idx: torch.Tensor, batch: int = 32768) -> dict[str, float]:
    model.eval()
    acc = None
    for s in range(0, len(idx), batch):
        inputs, target, ev, evmask, w, pot = data.batch(idx[s:s + batch])
        logp = torch.log_softmax(model(*inputs), dim=1)
        acc = _accumulate(acc, weighted_metrics(logp, target, ev, evmask, w, pot))
    model.train()
    return _finish(acc)


@torch.no_grad()
def baselines(data: Tensors) -> dict[str, dict[str, float]]:
    """Uniform-over-legal and per-(street, node type) average solver strategy."""
    out = {}
    # Node type: street, actor, facing, and the legal action set.
    legal = data.legal.long()
    code = data.numeric[:, 0].long() * 8 + data.numeric[:, 1].long() * 4 + data.numeric[:, 2].long() * 2
    key = code * 1024 + (legal * (2 ** torch.arange(legal.shape[1], device=legal.device))).sum(1)
    keys, inverse = torch.unique(key, return_inverse=True)
    sums = torch.zeros(len(keys), legal.shape[1], device=data.device)
    for s in range(0, len(data.train_idx), 1 << 20):
        idx = data.train_idx[s:s + (1 << 20)]
        node = data.s_node[idx]
        sums.index_add_(0, inverse[node], data.s_target[idx].float() * data.s_weight[idx, None])
    avg = sums / sums.sum(1, keepdim=True).clamp_min(1e-30)
    for name in ("uniform", "node_average"):
        acc = None
        for s in range(0, len(data.test_idx), 1 << 18):
            idx = data.test_idx[s:s + (1 << 18)]
            inputs, target, ev, evmask, w, pot = data.batch(idx)
            lg = inputs[5].float()
            if name == "uniform":
                p = lg / lg.sum(1, keepdim=True)
            else:
                p = avg[inverse[data.s_node[idx]]] * lg
                p = torch.where(p.sum(1, keepdim=True) > 0, p / p.sum(1, keepdim=True).clamp_min(1e-30),
                                lg / lg.sum(1, keepdim=True))
            logp = torch.where(p > 0, p.clamp_min(1e-12).log(), torch.full_like(p, float("-inf")))
            acc = _accumulate(acc, weighted_metrics(logp, target, ev, evmask, w, pot))
        out[name] = _finish(acc)
    return out


def _checkpoint(path: Path, model, opt, epoch: int, cfg: TrainConfig, history: list) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save({
        "model": model.state_dict(), "model_config": model.config, "optimizer": opt.state_dict(),
        "epoch": epoch, "config": asdict(cfg), "history": history,
        "rng": {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
    }, tmp)
    os.replace(tmp, path)  # never leave a half-written checkpoint


def train(cfg: TrainConfig, run_dir: Path, device: str | None = None, resume: bool = True,
          stop_after_epoch: int | None = None, log=print) -> dict[str, Any]:
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt = run_dir / "checkpoint.pt"
    _set_seeds(cfg.seed)
    data = Tensors(Path(cfg.features), device)
    model = PolicyNet(cfg.hidden, cfg.layers, cfg.range_dim, cfg.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    start, history, resumed_from = 0, [], None
    if resume and ckpt.exists():
        state = torch.load(ckpt, map_location=device, weights_only=False)
        if state["config"] != asdict(cfg):
            raise ValueError(f"{ckpt} was trained with a different config; use a new run directory")
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        start, history, resumed_from = state["epoch"], state["history"], state["epoch"]
        log(f"resumed from epoch {start}")
    steps_per_epoch = -(-len(data.train_idx) // cfg.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=cfg.epochs * steps_per_epoch,
                                                last_epoch=start * steps_per_epoch - 1 if start else -1)
    t0 = time.perf_counter()
    for epoch in range(start, cfg.epochs):
        gen = torch.Generator(device="cpu").manual_seed(cfg.seed * 1000 + epoch)
        perm = data.train_idx[torch.randperm(len(data.train_idx), generator=gen).to(device)]
        total = 0.0
        for s in range(0, len(perm), cfg.batch_size):
            inputs, target, _ev, _m, w, _pot = data.batch(perm[s:s + cfg.batch_size])
            logp = torch.log_softmax(model(*inputs), dim=1)
            ce = -(target * torch.where(target > 0, logp, torch.zeros_like(logp))).sum(1)
            loss = (w * ce).sum() / w.sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item()
        val = evaluate(model, data, data.val_idx)
        history.append({"epoch": epoch + 1, "train_loss": total / steps_per_epoch, **val})
        if val["kl_nats"] <= min(h["kl_nats"] for h in history):
            torch.save({"model": model.state_dict(), "model_config": model.config, "epoch": epoch + 1},
                       run_dir / "best.pt")
        log(f"epoch {epoch + 1}/{cfg.epochs}: train CE {total / steps_per_epoch:.4f}, "
            f"val KL {val['kl_nats']:.4f}, val EV regret {val['ev_regret_pct_pot']:.3f}% pot "
            f"({time.perf_counter() - t0:.0f}s)")
        _checkpoint(ckpt, model, opt, epoch + 1, cfg, history)
        if stop_after_epoch is not None and epoch + 1 >= stop_after_epoch and epoch + 1 < cfg.epochs:
            raise KeyboardInterrupt(f"stopped after epoch {epoch + 1} (simulated interruption)")
    best = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    final = evaluate(model, data, data.test_idx)
    report = {
        "config": asdict(cfg),
        "device": str(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda if device.type == "cuda" else None,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "train_samples": len(data.train_idx), "val_samples": len(data.val_idx),
        "test_samples": len(data.test_idx),
        "test_spots": int(len(set(data.node_spot[data.s_node[data.test_idx].cpu().numpy()]))),
        "best_epoch": best["epoch"],
        "history": history, "test": final, "baselines": baselines(data),
        "targets": TARGETS,
        "meets_targets": {k: final[k] <= v for k, v in TARGETS.items()},
        "resumed_from": resumed_from,
        "seconds": time.perf_counter() - t0,
    }
    (run_dir / "report.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    torch.save({"model": model.state_dict(), "model_config": model.config, "report": report},
               run_dir / "model.pt")
    return report
