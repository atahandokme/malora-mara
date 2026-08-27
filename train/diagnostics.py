"""
Shared training diagnostics for PEFT experiments.

Captures three artifacts per run, written under <output_dir>:
  1. efficiency.json        — params, memory, throughput (once + final)
  2. diagnostics.jsonl       — per-step metrics (loss, grad norms, gate stats)
  3. gate_maps/epoch_N.npz   — per-token λ tensors for gate analysis

Works with any adapted linear module that exposes .lora_A, .lora_B, and
optionally .gate and .magnitude (MaLoRA, MaDoRA, plain LoRA/DoRA).
"""

# --- repo root on sys.path (entry points live one level below it) ---
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "chunking", _ROOT / "mara", _ROOT / "train", _ROOT / "eval"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --------------------------------------------------------------------


import json
import time
import re
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from collections import defaultdict


MATRIX_TYPES = ("q_proj", "k_proj", "v_proj", "up_proj", "down_proj", "o_proj", "gate_proj")


def _module_matrix_type(name):
    for mt in MATRIX_TYPES:
        if mt in name:
            return mt
    return "other"


def _module_layer_idx(name):
    m = re.search(r"layers\.(\d+)", name)
    return int(m.group(1)) if m else -1


def _safe_norm(t):
    if t is None:
        return 0.0
    return float(t.detach().float().norm().item())


def _resolve_lora_linear(attr):
    """PEFT wraps lora_A/B in ModuleDict{name:Linear}; native modules use a plain Linear.

    Returns the first underlying Linear or None.
    """
    if attr is None:
        return None
    if isinstance(attr, nn.Linear):
        return attr
    if isinstance(attr, nn.ModuleDict):
        for v in attr.values():
            if isinstance(v, nn.Linear):
                return v
    return None


def _safe_mean(t):
    if t is None:
        return 0.0
    return float(t.detach().float().mean().item())


class DiagnosticsLogger:
    """Per-run diagnostics. Call at (start, every step, epoch boundaries, end)."""

    def __init__(self, output_dir, log_every=50, gate_capture_layers=None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "gate_maps").mkdir(exist_ok=True)
        self.log_every = log_every
        self.jsonl_path = self.output_dir / "diagnostics.jsonl"
        self.eff_path = self.output_dir / "efficiency.json"
        self._jsonl = open(self.jsonl_path, "w", buffering=1)

        self._adapted_modules = None       # cached list of (name, module) with lora_A/lora_B
        self._gate_modules = None          # cached list of (name, gate)
        self._magnitude_init = {}          # name -> magnitude snapshot at start
        self._epoch_start_time = None
        self._run_start_time = time.time()
        self._tokens_seen = 0
        self._cur_epoch = 0

        # Gate output capture (forward-hook-based, cleared each step)
        self._gate_capture = defaultdict(list)  # name -> [λ tensors since last flush]
        self._gate_hooks = []
        self._gate_capture_layers = gate_capture_layers  # None = all

    # ----- discovery / setup -----

    def _discover(self, model):
        if self._adapted_modules is not None:
            return
        adapted, gates = [], []
        for name, module in model.named_modules():
            if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                la = _resolve_lora_linear(module.lora_A)
                lb = _resolve_lora_linear(module.lora_B)
                if la is None or lb is None:
                    continue
                adapted.append((name, module, la, lb))
                if hasattr(module, "gate") and isinstance(module.gate, nn.Module):
                    gates.append((name, module.gate))
                if hasattr(module, "magnitude") and isinstance(module.magnitude, nn.Parameter):
                    self._magnitude_init[name] = module.magnitude.data.detach().clone()
                elif hasattr(module, "lora_magnitude_vector"):
                    mv = module.lora_magnitude_vector
                    mv_p = None
                    if isinstance(mv, nn.ModuleDict):
                        for v in mv.values():
                            for p in v.parameters():
                                mv_p = p
                                break
                    elif isinstance(mv, nn.Parameter):
                        mv_p = mv
                    if mv_p is not None:
                        self._magnitude_init[name] = mv_p.data.detach().clone()
        self._adapted_modules = adapted
        self._gate_modules = gates

    def register_gate_hooks(self, model):
        """Attach forward hooks to every gate module to capture λ outputs.

        Only keeps a rolling window since last flush; flushed at every log_step.
        """
        self._discover(model)
        for name, gate in self._gate_modules:
            def _hook(mod, inp, out, _name=name):
                # λ can be (B, S, 1) scalar or (B, S, r) diagonal
                self._gate_capture[_name].append(out.detach().float().cpu())
            h = gate.register_forward_hook(_hook)
            self._gate_hooks.append(h)

    def remove_gate_hooks(self):
        for h in self._gate_hooks:
            h.remove()
        self._gate_hooks = []

    # ----- efficiency snapshot -----

    def log_efficiency(self, model, optimizer, base_model_name="", config_dict=None):
        self._discover(model)
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

        gate_params = 0
        lora_params = 0
        mag_params = 0
        for name, module, la, lb in self._adapted_modules:
            lora_params += la.weight.numel() + lb.weight.numel()
            if hasattr(module, "magnitude") and isinstance(module.magnitude, nn.Parameter):
                mag_params += module.magnitude.numel()
            if hasattr(module, "gate") and isinstance(module.gate, nn.Module):
                for p in module.gate.parameters():
                    gate_params += p.numel()

        # Optimizer state memory estimate (Adam: 2x trainable for m and v)
        opt_mem_bytes = 0
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    opt_mem_bytes += 2 * p.numel() * p.element_size()

        try:
            peak_mem = torch.cuda.max_memory_allocated()
            cur_mem = torch.cuda.memory_allocated()
        except Exception:
            peak_mem, cur_mem = 0, 0

        eff = {
            "base_model": base_model_name,
            "total_params": total,
            "trainable_params": trainable,
            "trainable_ratio": trainable / max(total, 1),
            "lora_AB_params": lora_params,
            "gate_params": gate_params,
            "magnitude_params": mag_params,
            "num_adapted_modules": len(self._adapted_modules),
            "num_gate_modules": len(self._gate_modules),
            "optimizer_state_mb": opt_mem_bytes / 1024**2,
            "gpu_mem_allocated_mb": cur_mem / 1024**2,
            "gpu_mem_peak_mb": peak_mem / 1024**2,
            "config": config_dict or {},
        }
        with open(self.eff_path, "w") as f:
            json.dump(eff, f, indent=2)
        return eff

    # ----- per-step -----

    def log_step(self, step, loss, model, optimizer, batch_tokens=0,
                  task_loss=None, drift_loss=None):
        self._tokens_seen += batch_tokens
        if step % self.log_every != 0:
            # Drop captured gate outputs we won't summarize (keep memory low)
            self._gate_capture.clear()
            return None

        self._discover(model)
        row = {
            "epoch": self._cur_epoch,
            "step": step,
            "loss": float(loss.item()) if torch.is_tensor(loss) else float(loss),
            "time": time.time() - self._run_start_time,
            "tokens_seen": self._tokens_seen,
        }
        # Optional split-loss reporting (for state-drift-regularized runs)
        if task_loss is not None:
            row["loss_task"]  = float(task_loss.item()) if torch.is_tensor(task_loss) else float(task_loss)
        if drift_loss is not None:
            row["loss_drift"] = float(drift_loss.item()) if torch.is_tensor(drift_loss) else float(drift_loss)

        # --- gate output stats (from hooks) ---
        if self._gate_capture:
            all_lam = []
            per_layer = defaultdict(list)
            for name, lst in self._gate_capture.items():
                if not lst:
                    continue
                flat = torch.cat([t.reshape(-1) for t in lst])
                all_lam.append(flat)
                li = _module_layer_idx(name)
                per_layer[li].append(flat)
            if all_lam:
                lam = torch.cat(all_lam)
                row["gate/lambda_mean"] = float(lam.mean())
                row["gate/lambda_std"] = float(lam.std())
                row["gate/lambda_min"] = float(lam.min())
                row["gate/lambda_max"] = float(lam.max())
                # torch.quantile caps at ~16M elements; subsample for large tensors
                if lam.numel() > 1_000_000:
                    idx = torch.randint(0, lam.numel(), (1_000_000,))
                    sub = lam[idx]
                else:
                    sub = lam
                row["gate/lambda_p10"] = float(sub.quantile(0.1))
                row["gate/lambda_p90"] = float(sub.quantile(0.9))
                row["gate/lambda_mean_per_layer"] = {
                    str(li): float(torch.cat(v).mean()) for li, v in per_layer.items()
                }
        self._gate_capture.clear()

        # --- gate parameter norms & grads ---
        mamba_pn, mamba_gn = [], []
        bias_vals = []
        for name, gate in self._gate_modules:
            for pname, p in gate.named_parameters():
                if "mamba" in pname:
                    mamba_pn.append(_safe_norm(p))
                    if p.grad is not None:
                        mamba_gn.append(_safe_norm(p.grad))
                if pname.endswith("gate_bias"):
                    bias_vals.append(float(p.data.mean().item()))
        if mamba_pn:
            row["gate/mamba_param_norm_mean"] = sum(mamba_pn) / len(mamba_pn)
        if mamba_gn:
            row["gate/mamba_grad_norm_mean"] = sum(mamba_gn) / len(mamba_gn)
            row["gate/mamba_grad_norm_max"] = max(mamba_gn)
        if bias_vals:
            row["gate/bias_mean"] = sum(bias_vals) / len(bias_vals)

        # --- LoRA A/B gradient norms, split by matrix type ---
        a_grads = defaultdict(list)
        b_grads = defaultdict(list)
        mag_drift = []
        for name, module, la, lb in self._adapted_modules:
            mt = _module_matrix_type(name)
            if la.weight.grad is not None:
                a_grads[mt].append(_safe_norm(la.weight.grad))
            if lb.weight.grad is not None:
                b_grads[mt].append(_safe_norm(lb.weight.grad))
            if name in self._magnitude_init:
                cur = None
                if hasattr(module, "magnitude") and isinstance(module.magnitude, nn.Parameter):
                    cur = module.magnitude.data
                elif hasattr(module, "lora_magnitude_vector"):
                    mv = module.lora_magnitude_vector
                    if isinstance(mv, nn.ModuleDict):
                        for v in mv.values():
                            for p in v.parameters():
                                cur = p.data
                                break
                    elif isinstance(mv, nn.Parameter):
                        cur = mv.data
                if cur is not None:
                    delta = (cur - self._magnitude_init[name]).float()
                    mag_drift.append(float(delta.norm().item()))

        if a_grads:
            flat_a = [g for gs in a_grads.values() for g in gs]
            flat_b = [g for gs in b_grads.values() for g in gs]
            row["lora/A_grad_mean"] = sum(flat_a) / max(len(flat_a), 1)
            row["lora/B_grad_mean"] = sum(flat_b) / max(len(flat_b), 1)
            row["lora/A_grad_per_mtype"] = {
                k: sum(v) / len(v) for k, v in a_grads.items() if v
            }
            row["lora/B_grad_per_mtype"] = {
                k: sum(v) / len(v) for k, v in b_grads.items() if v
            }
        if mag_drift:
            row["magnitude/drift_mean"] = sum(mag_drift) / len(mag_drift)
            row["magnitude/drift_max"] = max(mag_drift)

        # --- memory / throughput ---
        try:
            row["gpu_mem_mb"] = torch.cuda.memory_allocated() / 1024**2
            row["gpu_mem_peak_mb"] = torch.cuda.max_memory_allocated() / 1024**2
        except Exception:
            pass
        if self._epoch_start_time is not None and self._tokens_seen > 0:
            elapsed = time.time() - self._epoch_start_time
            row["tokens_per_sec"] = self._tokens_seen / max(elapsed, 1e-6)

        self._jsonl.write(json.dumps(row) + "\n")
        return row

    # ----- epoch boundaries -----

    def start_epoch(self, epoch):
        self._cur_epoch = epoch
        self._epoch_start_time = time.time()
        self._tokens_seen = 0
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    def end_epoch(self, epoch, train_loss, val_loss, num_steps):
        wall = time.time() - self._epoch_start_time if self._epoch_start_time else 0.0
        try:
            peak = torch.cuda.max_memory_allocated() / 1024**2
        except Exception:
            peak = 0
        row = {
            "epoch": epoch,
            "event": "epoch_end",
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "num_steps": num_steps,
            "wall_time_sec": wall,
            "gpu_mem_peak_mb": peak,
            "tokens_per_sec": self._tokens_seen / max(wall, 1e-6),
        }
        self._jsonl.write(json.dumps(row) + "\n")
        return row

    # ----- gate snapshots -----

    @torch.no_grad()
    def snapshot_gate_outputs(self, model, batch, epoch, tokenizer=None, tag=""):
        """Run one batch forward, capture λ for every gate module, save to npz.

        Stores:
          - lambdas: dict[module_name] → (B, S, r_or_1) numpy
          - module_names: list[str]
          - input_ids: (B, S) numpy
          - labels: (B, S) numpy (for context/answer mask)
          - tokens: list[list[str]] (if tokenizer provided)
        """
        self._discover(model)
        was_training = model.training
        model.eval()
        # Temporary capture hooks
        captures = {}

        def _make_hook(n):
            def _h(mod, inp, out):
                captures[n] = out.detach().float().cpu().numpy()
            return _h

        handles = []
        for name, gate in self._gate_modules:
            handles.append(gate.register_forward_hook(_make_hook(name)))

        try:
            inp = {k: v.to(model.device) for k, v in batch.items() if torch.is_tensor(v)}
            _ = model(**inp)
        finally:
            for h in handles:
                h.remove()
            if was_training:
                model.train()

        out_path = self.output_dir / "gate_maps" / f"epoch_{epoch}{('_' + tag) if tag else ''}.npz"
        tokens_out = None
        if tokenizer is not None:
            tokens_out = np.array([
                [tokenizer.decode([tid]) for tid in row]
                for row in batch["input_ids"].cpu().numpy()
            ], dtype=object)

        save_dict = {
            "module_names": np.array([n for n, _ in self._gate_modules], dtype=object),
            "input_ids": batch["input_ids"].cpu().numpy(),
            "labels": batch["labels"].cpu().numpy() if "labels" in batch else np.array([]),
            "attention_mask": batch["attention_mask"].cpu().numpy() if "attention_mask" in batch else np.array([]),
        }
        for n, arr in captures.items():
            save_dict[f"lam__{n}"] = arr
        if tokens_out is not None:
            save_dict["tokens"] = tokens_out
        np.savez_compressed(out_path, **save_dict)
        return out_path

    # ----- end -----

    def finish(self):
        self.remove_gate_hooks()
        try:
            self._jsonl.close()
        except Exception:
            pass
