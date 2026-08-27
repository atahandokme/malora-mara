"""
Training script for MaLoRA.

SSM-modulated LoRA: a Mamba recurrence over the token sequence produces the
modulation factor λ_t that scales the low-rank update:
    h_t = W_0 x_t + (α/r) * B @ (λ_t ⊙ A @ x_t)

The paper configuration is `gate_type: mamba` with `scalar_output: true`,
giving a scalar λ_t broadcast over the r rank directions. `scalar_output: false`
gives λ_t ∈ R^r, one factor per direction, reported only as an ablation.

Supports layer splitting: gate_start_layer controls which layers get
input-dependent modulators vs fixed g=1 (standard LoRA).
"""

# --- repo root on sys.path (entry points live one level below it) ---
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "chunking", _ROOT / "mara", _ROOT / "train", _ROOT / "eval"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --------------------------------------------------------------------


import os
import yaml
import torch
import argparse
import time
import json
from pathlib import Path

# Fix memory fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Enable TF32 for A100 (~3x matmul speedup, negligible precision loss)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from tqdm import tqdm
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader, random_split

from model.malora import inject_malora, set_answer_mask, MaLoRALinear
from diagnostics import DiagnosticsLogger
from data.musique import MuSiQueDataset
from data.twowikimultihopqa import TwoWikiMultihopQADataset
from data.commonsense import CommonsenseDataset
from data.gsm8k import GSM8KDataset


def log_gate_diagnostics(model, step):
    """Log gate and gradient diagnostics for all MaLoRALinear modules."""
    gate_grads = []
    lora_a_grads = []
    lora_b_grads = []
    gate_bias_values = []

    for name, module in model.named_modules():
        if not isinstance(module, MaLoRALinear):
            continue

        # Gate bias values (what softplus sees as input)
        if hasattr(module.gate, 'gate_bias'):
            bias = module.gate.gate_bias.data
            gate_bias_values.append(bias)

        # Gradient norms
        if hasattr(module.gate, 'gate_bias') and module.gate.gate_bias.grad is not None:
            gate_grads.append(module.gate.gate_bias.grad.norm().item())
        if module.lora_A.weight.grad is not None:
            lora_a_grads.append(module.lora_A.weight.grad.norm().item())
        if module.lora_B.weight.grad is not None:
            lora_b_grads.append(module.lora_B.weight.grad.norm().item())

    if not gate_bias_values:
        return

    # Compute softplus of gate biases to show actual λ init/current values
    import torch.nn.functional as F
    all_biases = torch.cat(gate_bias_values)
    all_lambdas = F.softplus(all_biases)

    print(f"\n  [Step {step}] Gate diagnostics:", flush=True)
    print(f"    Gate λ (softplus of bias): mean={all_lambdas.mean():.4f}, std={all_lambdas.std():.4f}, "
          f"min={all_lambdas.min():.4f}, max={all_lambdas.max():.4f}", flush=True)
    print(f"    Gate bias raw: mean={all_biases.mean():.4f}, std={all_biases.std():.4f}", flush=True)
    if gate_grads and lora_a_grads and lora_b_grads:
        print(f"    Grad norms — gate_bias: {sum(gate_grads)/len(gate_grads):.6f}, "
              f"lora_A: {sum(lora_a_grads)/len(lora_a_grads):.6f}, "
              f"lora_B: {sum(lora_b_grads)/len(lora_b_grads):.6f}", flush=True)
    else:
        print(f"    Grad norms — not available (grads may have been zeroed)", flush=True)

    # Log Mamba SSM parameter norms and grad norms
    mamba_param_norms = []
    mamba_grad_norms = []
    # Linear gate's `down` projection (the L2 per-token path in chunk_residual_mamba
    # and the input projection in chunk_mamba / hybrid_sigmoidA). Tracks whether
    # the linear path is getting gradient signal, complement to Mamba grads above.
    down_param_norms = []
    down_grad_norms = []
    for name, module in model.named_modules():
        if not isinstance(module, MaLoRALinear):
            continue
        if hasattr(module.gate, 'mamba'):
            for pname, p in module.gate.mamba.named_parameters():
                mamba_param_norms.append(p.data.float().norm().item())
                if p.grad is not None:
                    mamba_grad_norms.append(p.grad.float().norm().item())
        # `down` weight — present on chunk_mamba, hybrid_sigmoidA, chunk_residual
        if hasattr(module.gate, 'down') and module.gate.down is not None:
            p = module.gate.down.weight
            down_param_norms.append(p.data.float().norm().item())
            if p.grad is not None:
                down_grad_norms.append(p.grad.float().norm().item())

    if mamba_param_norms:
        print(f"    Mamba SSM params: norm_mean={sum(mamba_param_norms)/len(mamba_param_norms):.6f}, "
              f"n_params={len(mamba_param_norms)}", flush=True)
    if mamba_grad_norms:
        print(f"    Mamba SSM grads:  norm_mean={sum(mamba_grad_norms)/len(mamba_grad_norms):.6f}, "
              f"norm_max={max(mamba_grad_norms):.6f}", flush=True)
    if down_param_norms:
        print(f"    Linear `down` params: norm_mean={sum(down_param_norms)/len(down_param_norms):.6f}, "
              f"n_modules={len(down_param_norms)}", flush=True)
    if down_grad_norms:
        print(f"    Linear `down` grads:  norm_mean={sum(down_grad_norms)/len(down_grad_norms):.6f}, "
              f"norm_max={max(down_grad_norms):.6f}", flush=True)

    # Hybrid-gate α (residual_outer): track mixing-weight distribution over modules.
    # With sigmoid_alpha=True, alpha stores the logit g; effective mix = σ(g).
    alpha_raw = []
    alpha_grads = []
    sigmoid_flag = False
    for name, module in model.named_modules():
        if not isinstance(module, MaLoRALinear):
            continue
        if hasattr(module.gate, 'alpha') and module.gate.alpha is not None:
            alpha_raw.append(module.gate.alpha.data.float().item())
            if module.gate.alpha.grad is not None:
                alpha_grads.append(module.gate.alpha.grad.float().abs().item())
            if getattr(module.gate, 'sigmoid_alpha', False):
                sigmoid_flag = True
    if alpha_raw:
        import numpy as _np
        raw = _np.array(alpha_raw)
        # Always show σ(raw) too — interpretable for sigmoid_alpha=True gates,
        # and for direct-α gates it's just an extra sigmoid of the value (still
        # informative as a normalized 0..1 view, no harm).
        eff = 1.0 / (1.0 + _np.exp(-raw))
        kind = "logit" if sigmoid_flag else "scalar"
        print(f"    α (raw {kind}): mean={raw.mean():+.4f}, std={raw.std():.4f}, "
              f"min={raw.min():+.4f}, max={raw.max():+.4f}", flush=True)
        print(f"    α (σ effective): mean={eff.mean():.4f}, std={eff.std():.4f}, "
              f"min={eff.min():.4f}, max={eff.max():.4f}", flush=True)
        if alpha_grads:
            g = _np.array(alpha_grads)
            print(f"    α grad |mean|: {g.mean():.6f}, |max|: {g.max():.6f}", flush=True)


def _log_chunk_gate_trace(model, epoch, step, diag=None, chunk_names_batch=None):
    """Dump per-chunk gates from every ChunkMambaGate to chunk_gate_trace.jsonl.

    Produces one JSON line per (module, batch_item) with mean-over-rank gate
    values per chunk. Also prints a summary to stdout.
    """
    import json as _json
    from pathlib import Path as _Path
    from model.chunk_mamba_gate import (
        snapshot_chunk_gates, summarize_chunk_gates,
    )
    snap = snapshot_chunk_gates(model, reduce="mean_rank")   # {name: (B, K)}
    if not snap:
        return
    records = summarize_chunk_gates(snap, chunk_names_batch)

    # Aggregate stats for stdout
    all_means = [sum(r["gate_means"]) / max(1, len(r["gate_means"])) for r in records]
    all_std = [r["std_across_chunks"] for r in records]
    overall_mean = sum(all_means) / len(all_means) if all_means else 0.0
    mean_std_across_chunks = sum(all_std) / len(all_std) if all_std else 0.0
    max_std = max(all_std) if all_std else 0.0
    print(f"    [chunk-gate] modules={len(snap)}  "
          f"μ̄(gate)={overall_mean:.3f}  "
          f"avg σ across chunks={mean_std_across_chunks:.4f}  "
          f"max σ across chunks={max_std:.4f}", flush=True)

    # Write to disk if we have an output_dir
    if diag is None or not hasattr(diag, "output_dir"):
        return
    out_path = _Path(diag.output_dir) / "chunk_gate_trace.jsonl"
    with open(out_path, "a") as f:
        for r in records:
            r["epoch"] = epoch
            r["step"] = step
            f.write(_json.dumps(r) + "\n")


def run_gate_analysis_on_eval(model, tokenizer, dataset_name, config, n_samples=100):
    """Run gate analysis during eval: capture live gate values across samples."""
    import torch.nn.functional as F
    from collections import defaultdict

    # Hook to capture gate outputs
    captured_gates = {}
    hooks = []

    def make_hook(name):
        def hook_fn(module, input, output):
            # output is base_out + lora_out, but we need λ
            # Re-extract λ from the gate
            x = input[0]
            if hasattr(module.gate, 'direct_mode') and module.gate.direct_mode:
                Ax = module.lora_A(x)
                lam = module.gate(x, Ax=Ax)
            else:
                lam = module.gate(x)
            captured_gates[name] = lam.detach()
        return hook_fn

    for name, module in model.named_modules():
        if isinstance(module, MaLoRALinear):
            hooks.append(module.register_forward_hook(make_hook(name)))

    # Collect stats
    layer_stats = defaultdict(lambda: {'sum': 0, 'sum_sq': 0, 'min': float('inf'),
                                        'max': float('-inf'), 'count': 0})

    if dataset_name == 'quality':
        from datasets import load_dataset
        from data.quality import format_quality_prompt
        eval_dataset = load_dataset('emozilla/quality', split='validation')
    elif dataset_name == 'hotpotqa':
        from datasets import load_dataset
        eval_dataset = load_dataset('hotpot_qa', 'fullwiki', split='validation')
    else:
        print(f"  Gate analysis not supported for {dataset_name}")
        for h in hooks:
            h.remove()
        return

    n_samples = min(n_samples, len(eval_dataset))
    print(f"\n  Running gate analysis on {n_samples} eval samples...")

    with torch.no_grad():
        for i in range(n_samples):
            item = eval_dataset[i]
            if dataset_name == 'quality':
                prompt = format_quality_prompt(item['article'], item['question'],
                                               item['options'], answer_idx=None)
            else:
                context = ' '.join(item['context']['sentences'][0]) if item['context']['sentences'] else ''
                prompt = f"Context: {context}\n\nQuestion: {item['question']}\n\nAnswer:"

            inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=8192)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            try:
                model(**inputs)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                continue

            for name, lam in captured_gates.items():
                # lam: (1, seq, r) or (1, seq, 1)
                vals = lam.float().squeeze(0)  # (seq, r)
                mean_per_token = vals.mean(dim=-1)  # (seq,)
                layer_stats[name]['sum'] += mean_per_token.sum().item()
                layer_stats[name]['sum_sq'] += (mean_per_token ** 2).sum().item()
                layer_stats[name]['min'] = min(layer_stats[name]['min'], mean_per_token.min().item())
                layer_stats[name]['max'] = max(layer_stats[name]['max'], mean_per_token.max().item())
                layer_stats[name]['count'] += mean_per_token.numel()

            if (i + 1) % 50 == 0:
                print(f"    Gate analysis: {i+1}/{n_samples} samples processed")

    # Remove hooks
    for h in hooks:
        h.remove()

    # Print summary
    print(f"\n  Gate Analysis Summary ({n_samples} samples):")
    print(f"  {'Module':<60} {'Mean':>7} {'Std':>7} {'Min':>7} {'Max':>7} {'Range':>7}")
    print(f"  {'-'*95}")
    for name in sorted(layer_stats.keys()):
        s = layer_stats[name]
        n = s['count']
        mean = s['sum'] / n
        std = ((s['sum_sq'] / n) - mean**2) ** 0.5
        print(f"  {name:<60} {mean:7.4f} {std:7.4f} {s['min']:7.4f} {s['max']:7.4f} {s['max']-s['min']:7.4f}")


def mezo_gate_step(model, batch, gate_modules, mezo_lr=1e-3, eps=1e-3):
    """
    Zeroth-order (MeZO/SPSA) optimization step for gate parameters only.
    Bypasses the Ax gradient bottleneck by estimating gradients with
    two forward passes instead of backprop.

    grad ≈ (L(θ+εz) - L(θ-εz)) / (2ε) · z
    """
    # Collect all gate parameters
    gate_params = []
    for gate in gate_modules:
        for p in gate.parameters():
            gate_params.append(p)

    if not gate_params:
        return 0.0

    # Sample random perturbation z for each gate param (using shared seed for memory efficiency)
    seed = torch.randint(0, 2**31, (1,)).item()

    # Perturb +εz
    torch.manual_seed(seed)
    for p in gate_params:
        z = torch.randn_like(p.data)
        p.data.add_(eps * z)

    # Forward pass with +εz
    with torch.no_grad():
        outputs_plus = model(**batch)
        loss_plus = outputs_plus.loss.item()

    # Perturb -2εz (go from +εz to -εz)
    torch.manual_seed(seed)
    for p in gate_params:
        z = torch.randn_like(p.data)
        p.data.add_(-2 * eps * z)

    # Forward pass with -εz
    with torch.no_grad():
        outputs_minus = model(**batch)
        loss_minus = outputs_minus.loss.item()

    # Restore original params (+εz to get back to original)
    torch.manual_seed(seed)
    for p in gate_params:
        z = torch.randn_like(p.data)
        p.data.add_(eps * z)

    # Compute grad estimate and update
    proj_grad = (loss_plus - loss_minus) / (2 * eps)

    torch.manual_seed(seed)
    for p in gate_params:
        z = torch.randn_like(p.data)
        p.data.add_(-mezo_lr * proj_grad * z)

    return proj_grad


def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train_epoch(model, dataloader, optimizer, config, epoch, scheduler=None, diag=None,
                gate_params=None, lora_params=None, a_log_params=None,
                gate_modules=None):
    """Train for one epoch."""
    model.train()

    total_loss = 0
    num_batches = 0

    grad_accum_steps = config['training']['gradient_accumulation_steps']
    gate_grad_clip = config['training'].get('gate_grad_clip', None)
    a_log_grad_clip = config['training'].get('mamba_a_log_grad_clip', None)
    gate_entropy_weight = config['training'].get('gate_entropy_weight', 0.0)  # GateRA-style near-binary reg
    # State-engagement regularizer: REWARD A_log drift from init.
    # Loss = task_loss - state_drift_weight * mean(||A_log - A_log_init||^2)
    state_drift_weight = config['training'].get('state_drift_weight', 0.0)

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    skipped_oom = 0

    for batch_idx, batch in enumerate(pbar):
        # Extract non-tensor extras (chunk_ends is a list of lists, not a tensor)
        chunk_ends = batch.pop('chunk_ends', None)
        chunk_names_batch = batch.pop('chunk_names', None)
        batch = {k: v.to(model.device) for k, v in batch.items()}

        # If using chunk_mamba gates, push chunk boundaries to every gate
        if chunk_ends is not None:
            from model.chunk_mamba_gate import (
                set_chunk_ends_on_model,
            )
            set_chunk_ends_on_model(model, chunk_ends)

        try:
            # Mask gate at answer positions so gate only learns on context/question
            if config.get('gated_lora', {}).get('mask_answer_gates', False):
                set_answer_mask(model, batch.get('labels'))

            outputs = model(**batch)
            task_loss = outputs.loss / grad_accum_steps

            # Mamba state-engagement regularizer: subtract drift bonus from loss
            # so the optimizer is REWARDED for moving A_log away from its init.
            drift_bonus = None
            if state_drift_weight != 0.0 and gate_modules is not None:
                drift_terms = []
                for gate in gate_modules:
                    if hasattr(gate, "state_drift_loss"):
                        drift_terms.append(gate.state_drift_loss())
                if drift_terms:
                    drift_bonus = torch.stack(drift_terms).mean()
                    # subtract because we want to MAXIMIZE drift
                    loss = task_loss - state_drift_weight * drift_bonus / grad_accum_steps
                else:
                    loss = task_loss
            else:
                loss = task_loss

            # GateRA-style entropy regularization: push scalar sigmoid gate toward binary.
            # g in (0,1) treated as Bernoulli prob; penalize H(g) = -[g log g + (1-g) log(1-g)].
            if gate_entropy_weight != 0.0 and gate_modules is not None:
                ent_terms = []
                for gate in gate_modules:
                    g = getattr(gate, "_last_gate", None)
                    if g is not None:
                        gc = g.float().clamp(1e-6, 1 - 1e-6)
                        ent_terms.append((-(gc * gc.log() + (1 - gc) * (1 - gc).log())).mean())
                if ent_terms:
                    loss = loss + gate_entropy_weight * torch.stack(ent_terms).mean() / grad_accum_steps

            if batch_idx == 0:
                valid_labels = batch['labels'][batch['labels'] != -100]
                print(f"\n  First batch stats:")
                print(f"    Task loss: {task_loss.item() * grad_accum_steps:.4f}")
                if drift_bonus is not None:
                    print(f"    A_log drift bonus: {drift_bonus.item():.6f}  (× weight {state_drift_weight})")
                print(f"    Sequence length: {batch['input_ids'].shape[1]}")
                print(f"    Valid labels: {len(valid_labels)} tokens")
                print(f"    Logits range: [{outputs.logits.min().item():.2f}, {outputs.logits.max().item():.2f}]")

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n  NaN/Inf detected at batch {batch_idx}, skipping...")
                continue

            loss.backward()

            # Clear answer mask after backward pass
            if config.get('gated_lora', {}).get('mask_answer_gates', False):
                set_answer_mask(model, None)

            # Log gate diagnostics periodically
            log_every = config.get('logging', {}).get('log_every_n_steps', 50)
            if (batch_idx + 1) % log_every == 0:
                log_gate_diagnostics(model, batch_idx + 1)
                # Chunk-MaLoRA: also dump per-chunk gates from every module
                if chunk_ends is not None:
                    _log_chunk_gate_trace(model, epoch, batch_idx + 1,
                                          diag, chunk_names_batch)

            if diag is not None:
                ntok = int(batch['attention_mask'].sum().item())
                # Pass task_loss + drift_loss separately so we can see them in jsonl
                tl = task_loss.item() * grad_accum_steps if 'task_loss' in dir() else None
                dl = drift_bonus.item() if drift_bonus is not None else None
                diag.log_step(batch_idx, loss.item() * grad_accum_steps, model, optimizer,
                              batch_tokens=ntok, task_loss=tl, drift_loss=dl)

            if (batch_idx + 1) % grad_accum_steps == 0:
                # Amplify gate gradients before clipping (compensates for Ax bottleneck)
                gate_grad_scale = config.get('gated_lora', {}).get('gate_grad_scale', 1.0)
                if gate_grad_scale != 1.0:
                    for module in model.modules():
                        if isinstance(module, MaLoRALinear):
                            for p in module.gate.parameters():
                                if p.grad is not None:
                                    p.grad *= gate_grad_scale

                # Tight per-tensor clip on A_log to kill gradient spikes (high-variance signal)
                if a_log_grad_clip is not None and a_log_params:
                    torch.nn.utils.clip_grad_norm_(a_log_params, a_log_grad_clip)

                if gate_grad_clip is not None and gate_params is not None and lora_params is not None:
                    # Per-group grad clip: tight on gate, loose on lora (Ax-bottleneck safety net)
                    torch.nn.utils.clip_grad_norm_(gate_params, gate_grad_clip)
                    torch.nn.utils.clip_grad_norm_(lora_params, config['training']['max_grad_norm'])
                else:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        config['training']['max_grad_norm']
                    )
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item() * grad_accum_steps

        except torch.cuda.OutOfMemoryError:
            skipped_oom += 1
            print(f"\n  OOM at batch {batch_idx} (seq_len={batch['input_ids'].shape[1]}), skipping... ({skipped_oom} total skipped)")
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            continue
        num_batches += 1

        pbar.set_postfix({'loss': f"{loss.item() * grad_accum_steps:.4f}"})

    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    if skipped_oom > 0:
        print(f"  Skipped {skipped_oom} OOM batches this epoch")
    return avg_loss, num_batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='Path to gated LoRA config YAML')
    parser.add_argument('--val-split', type=float, default=0.05)
    parser.add_argument('--max-train-samples', type=int, default=None)
    parser.add_argument('--lora-pool-path', type=str, default=None,
                        help='Path to canonical 10k JSON; if set, dataset loads from this instead of HF + filter')
    parser.add_argument('--output-root', type=str, default=None,
                        help='Override output directory root (e.g., outputs/canonical_pool)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed for data split + init + shuffle. '
                             'Embedded as _seedNN in output dir name when != 42.')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    gate_type = config['gated_lora']['gate_type']
    print(f"Loaded config from {args.config}")
    print(f"Gate type: {gate_type}")

    # Create output directory with descriptive name
    dataset_name = config.get('dataset', {}).get('name', 'hotpotqa')
    gate_sharing = config['gated_lora'].get('gate_sharing', 'per_layer')
    rank = config.get('gated_lora', {}).get('rank', config.get('lora', {}).get('rank', 16))
    activation = config['gated_lora'].get('gate_activation', 'softplus')
    bn = config['gated_lora'].get('gate_bottleneck', 32)
    d_state = config['gated_lora'].get('d_state', 16)
    gate_start = config['gated_lora'].get('gate_start_layer', 0)
    # Short model name: "Qwen/Qwen2.5-7B-Instruct" -> "qwen2.5-7b"
    base_model = config['model']['base_model']
    model_short = base_model.split('/')[-1].lower().replace('-instruct', '').replace('-it', '')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    start_tag = f"_start{gate_start}" if gate_start > 0 else ""
    direct_mode = config['gated_lora'].get('direct_mode', False)
    hproj_mode = config['gated_lora'].get('hproj_mode', False)
    scalar_output = config['gated_lora'].get('scalar_output', False)
    mode_tag = "direct" if direct_mode else ("hproj" if hproj_mode else "bn")
    output_tag = "scalar" if scalar_output else "diag"
    run_suffix = config.get('run_tag') or config.get('output_dir_suffix')
    suffix_str = f"_{run_suffix}" if run_suffix else ""
    seed_tag = "" if args.seed == 42 else f"_seed{args.seed}"
    output_root = Path(args.output_root or config.get('output_root', 'outputs'))
    output_dir = output_root / f"{dataset_name}_{mode_tag}_{output_tag}_r{rank}_d{d_state}{start_tag}_{activation}_{model_short}{suffix_str}{seed_tag}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Save config copy
    with open(output_dir / "config.yaml", 'w') as f:
        yaml.dump(config, f)

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config['model']['base_model'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    max_samples = args.max_train_samples if args.max_train_samples is not None else config['training'].get('max_train_samples')

    lora_pool_path = args.lora_pool_path or config.get('lora_pool_path')
    if dataset_name == 'musique':
        # If the gate is a chunk variant, request chunk-aware tokenization.
        use_chunks_mq = config['gated_lora'].get('gate_type') in (
            'chunk_mamba', 'chunk_token_mamba', 'chunk_residual_mamba'
        )
        print(f"\nLoading MuSiQue dataset (with_chunks={use_chunks_mq})...")
        full_dataset = MuSiQueDataset(
            tokenizer=tokenizer,
            max_length=config['training']['max_seq_length'],
            split='train',
            max_samples=max_samples,
            with_chunks=use_chunks_mq,
            lora_pool_path=lora_pool_path,
        )
    elif dataset_name == 'twowikimultihopqa':
        print("\nLoading 2WikiMultihopQA dataset...")
        full_dataset = TwoWikiMultihopQADataset(
            tokenizer=tokenizer,
            max_length=config['training']['max_seq_length'],
            split='train',
            max_samples=max_samples,
            lora_pool_path=lora_pool_path,
        )
    elif dataset_name == 'qasper':
        print("\nLoading Qasper dataset...")
        from data.qasper import QasperDataset
        full_dataset = QasperDataset(
            tokenizer=tokenizer,
            max_length=config['training']['max_seq_length'],
            split='train',
            max_samples=max_samples,
        )
    elif dataset_name == 'drop':
        print("\nLoading DROP dataset...")
        from data.drop import DROPDataset
        full_dataset = DROPDataset(
            tokenizer=tokenizer,
            max_length=config['training']['max_seq_length'],
            split='train',
            max_samples=max_samples,
            use_chat_template=False,
        )
    elif dataset_name == 'quality':
        question_first = config.get('dataset', {}).get('question_first', False)
        print("\nLoading QuALITY dataset...")
        from data.quality import QuALITYDataset
        full_dataset = QuALITYDataset(
            tokenizer=tokenizer,
            max_length=config['training']['max_seq_length'],
            split='train',
            max_samples=max_samples,
            question_first=question_first,
        )
    elif dataset_name == 'commonsense':
        print("\nLoading Commonsense 170k dataset...")
        full_dataset = CommonsenseDataset(
            tokenizer=tokenizer,
            max_length=config['training']['max_seq_length'],
            max_samples=max_samples,
        )
    elif dataset_name == 'gsm8k':
        print("\nLoading GSM-8K dataset...")
        full_dataset = GSM8KDataset(
            tokenizer=tokenizer,
            max_length=config['training']['max_seq_length'],
            split='train',
        )
    elif dataset_name == 'law':
        print("\nLoading Lawyer-Instruct dataset...")
        from data.law import LawDataset
        full_dataset = LawDataset(
            tokenizer=tokenizer,
            max_length=config['training']['max_seq_length'],
            split='train',
            max_samples=max_samples,
        )
    elif dataset_name == 'mixture':
        print("\nLoading 4-task mixture (MuSiQue + GSM8K + Law + Commonsense)...")
        samples_per_task = config.get('dataset', {}).get('samples_per_task', 1000)
        from data.mixture import MixtureDataset
        full_dataset = MixtureDataset(
            tokenizer=tokenizer,
            max_length=config['training']['max_seq_length'],
            samples_per_task=samples_per_task,
            seed=args.seed,
        )
    elif dataset_name == 'govreport':
        summary_reserve = config.get('dataset', {}).get('summary_reserve', 1024)
        use_filtered = config.get('dataset', {}).get('use_filtered', False)
        print("\nLoading GovReport long-context summarization dataset...")
        from data.govreport import GovReportDataset
        full_dataset = GovReportDataset(
            tokenizer=tokenizer,
            max_length=config['training']['max_seq_length'],
            split='train',
            max_samples=max_samples,
            summary_reserve=summary_reserve,
            use_filtered=use_filtered,
        )
    elif dataset_name == 'narrativeqa':
        gold_idx = config.get('dataset', {}).get('gold_answer_index', 0)
        data_subset = config.get('dataset', {}).get('subset', '30k')
        print(f"\nLoading NarrativeQA long-context book/script QA dataset (subset={data_subset})...")
        from data.narrativeqa import NarrativeQADataset
        full_dataset = NarrativeQADataset(
            tokenizer=tokenizer,
            max_length=config['training']['max_seq_length'],
            split='train',
            max_samples=max_samples,
            gold_answer_index=gold_idx,
            data_subset=data_subset,
        )
    else:
        # If the gate is chunk_mamba / chunk_token_mamba / chunk_residual_mamba,
        # we need per-example chunk boundaries. with_chunks=True enables
        # offset-based tokenization + chunk detection (and AMASK truncation of A).
        use_chunks = config['gated_lora'].get('gate_type') in (
            'chunk_mamba', 'chunk_token_mamba', 'chunk_residual_mamba'
        )
        print(f"\nLoading HotpotQA dataset (with_chunks={use_chunks}, lora_pool_path={lora_pool_path})...")
        from data.hotpotqa import HotpotQADataset
        full_dataset = HotpotQADataset(
            tokenizer=tokenizer,
            max_length=config['training']['max_seq_length'],
            split='train',
            max_samples=max_samples,
            with_chunks=use_chunks,
            lora_pool_path=lora_pool_path,
        )

    # Split into train/val. Seed controls split + init + shuffle.
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size

    # Seed torch/numpy/python so init + dropout differ across seeds
    import random as _py_random, numpy as _np
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    _np.random.seed(args.seed)
    _py_random.seed(args.seed)

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")

    # Collate function for dynamic padding (same as other training scripts)
    def collate_fn(batch):
        max_len = max(len(item['input_ids']) for item in batch)

        input_ids = []
        attention_mask = []
        labels = []

        for item in batch:
            seq_len = len(item['input_ids'])
            padding_len = max_len - seq_len

            input_ids.append(torch.cat([
                item['input_ids'],
                torch.full((padding_len,), tokenizer.pad_token_id, dtype=item['input_ids'].dtype)
            ]))

            attention_mask.append(torch.cat([
                item['attention_mask'],
                torch.zeros(padding_len, dtype=item['attention_mask'].dtype)
            ]))

            labels.append(torch.cat([
                item['labels'],
                torch.full((padding_len,), -100, dtype=item['labels'].dtype)
            ]))

        out = {
            'input_ids': torch.stack(input_ids),
            'attention_mask': torch.stack(attention_mask),
            'labels': torch.stack(labels),
        }
        # Pass through non-tensor extras (chunk_ends is a List[List[int]])
        if 'chunk_ends' in batch[0]:
            out['chunk_ends'] = [item['chunk_ends'] for item in batch]
            out['chunk_names'] = [item.get('chunk_names', []) for item in batch]
        return out

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # Load model
    print("\nLoading base model...")
    torch_dtype = torch.bfloat16 if config['model']['torch_dtype'] == 'bfloat16' else torch.float32

    # Use explicit single-GPU placement (not device_map='auto').
    # 'auto' triggers accelerate's offload-to-CPU heuristic, leaving some
    # params on `meta` device, which crashes when stacked-Mamba gates
    # (n_layers>=2) introduce extra modules that interact with offloaded weights.
    model = AutoModelForCausalLM.from_pretrained(
        config['model']['base_model'],
        torch_dtype=torch_dtype,
        device_map={'': 'cuda:0'},
        attn_implementation="sdpa",
    )

    # Inject MaLoRA
    print("\nInjecting MaLoRA...")
    gated_lora_config = {
        'gate_type': config['gated_lora']['gate_type'],
        'rank': config['gated_lora']['rank'],
        'alpha': config['gated_lora']['alpha'],
        'gate_bottleneck': config['gated_lora'].get('gate_bottleneck', 32),
        'gate_activation': config['gated_lora'].get('gate_activation', 'softplus'),
        'gate_sharing': config['gated_lora'].get('gate_sharing', 'per_matrix'),
        'target_modules': config['gated_lora'].get('target_modules', ['q_proj', 'k_proj', 'v_proj', 'o_proj']),
        'gate_start_layer': config['gated_lora'].get('gate_start_layer', 0),
        'd_state': config['gated_lora'].get('d_state', 16),
        'd_conv': config['gated_lora'].get('d_conv', 4),
        'expand_factor': config['gated_lora'].get('expand_factor', 2),
        'direct_mode': config['gated_lora'].get('direct_mode', False),
        'scalar_output': config['gated_lora'].get('scalar_output', False),
        'hproj_mode': config['gated_lora'].get('hproj_mode', False),
        'dropout': config['gated_lora'].get('dropout', 0.0),
        'inject_layers': config['gated_lora'].get('inject_layers', None),
        # Previously silently dropped — now correctly forwarded to the gate module.
        'n_layers': config['gated_lora'].get('n_layers', 1),
        'residual_outer': config['gated_lora'].get('residual_outer', False),
        'alpha_init': config['gated_lora'].get('alpha_init', 0.1),
        'zero_init_down': config['gated_lora'].get('zero_init_down', False),
        'freeze_bias': config['gated_lora'].get('freeze_bias', False),
        'binary_mode': config['gated_lora'].get('binary_mode', False),
        'binary_threshold': config['gated_lora'].get('binary_threshold', 0.5),
        'gate_clamp_max': config['gated_lora'].get('gate_clamp_max', None),
        'gate_scale': config['gated_lora'].get('gate_scale', 2.0),
    }

    model, gate_modules = inject_malora(model, gated_lora_config)

    # Convert trainable params to model dtype, then fix Mamba to float32
    model_dtype = torch_dtype
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.to(dtype=model_dtype)

    # SSM blocks must stay in float32 (pscan requires it).
    # Small scalar gate params (gate_bias, alpha, outer_ln) must also stay in
    # float32 — at their magnitude (~0.54, ~1.0) Adam updates of ~2e-4/step are
    # below the bf16 ULP and get silently rounded to zero, freezing the param.
    from model.malora import MambaModulator, LinearModulator, MLPModulator, RNNModulator
    from model.chunk_mamba_gate import (
        ChunkMambaGate, ChunkTokenMambaGate, ChunkResidualMambaGate,
    )
    for gate in gate_modules:
        if isinstance(gate, MambaModulator):
            gate.mamba = gate.mamba.float()
            if hasattr(gate, 'gate_bias') and gate.gate_bias is not None:
                gate.gate_bias.data = gate.gate_bias.data.float()
            if getattr(gate, 'outer_ln', None) is not None:
                gate.outer_ln = gate.outer_ln.float()
            if getattr(gate, 'alpha', None) is not None:
                gate.alpha.data = gate.alpha.data.float()
        elif isinstance(gate, (ChunkMambaGate, ChunkTokenMambaGate)):
            # Same treatment as MambaModulator: Mamba fp32, gate_bias fp32.
            gate.mamba = gate.mamba.float()
            if hasattr(gate, 'gate_bias') and gate.gate_bias is not None:
                gate.gate_bias.data = gate.gate_bias.data.float()
        elif isinstance(gate, ChunkResidualMambaGate):
            # Mamba fp32, LayerNorm fp32, gate_bias fp32, alpha fp32.
            gate.mamba = gate.mamba.float()
            if hasattr(gate, 'gate_bias') and gate.gate_bias is not None:
                gate.gate_bias.data = gate.gate_bias.data.float()
            if hasattr(gate, 'outer_ln') and gate.outer_ln is not None:
                gate.outer_ln = gate.outer_ln.float()
            if hasattr(gate, 'alpha') and gate.alpha is not None:
                gate.alpha.data = gate.alpha.data.float()
        elif isinstance(gate, RNNModulator):
            # Same treatment as MambaModulator: recurrence fp32 (cuDNN RNNs
            # reject mixed dtypes), gate_bias fp32 (bf16 ULP freezes it).
            gate.rnn = gate.rnn.float()
            if hasattr(gate, 'gate_bias') and gate.gate_bias is not None:
                gate.gate_bias.data = gate.gate_bias.data.float()
        elif isinstance(gate, LinearModulator):
            if gate.linear.bias is not None:
                gate.linear.bias.data = gate.linear.bias.data.float()
        elif isinstance(gate, MLPModulator):
            # MLP gate has biases at softplus-init magnitude; same fix
            for name, p in gate.named_parameters():
                if 'bias' in name:
                    p.data = p.data.float()

    # Enable gradient checkpointing
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    print(f"  Enabled gradient checkpointing")

    # Setup optimizer (only trainable params: LoRA A/B + gate)
    # Separate param groups if gate_lr is specified
    gate_lr = config['training'].get('gate_lr', None)
    alpha_lr_multiplier = config['training'].get('alpha_lr_multiplier', 1.0)
    a_log_lr_mult = config['training'].get('mamba_a_log_lr_multiplier', 1.0)
    lora_params = []
    gate_params_list = []
    alpha_params = []  # residual-outer α (scalar); separated when alpha_lr_multiplier != 1.0
    a_log_params = []  # mamba A_log; separated when mamba_a_log_lr_multiplier != 1.0
    gate_param_ids = set()
    for gate in gate_modules:
        for pname, p in gate.named_parameters():
            if not p.requires_grad:
                continue
            gate_param_ids.add(id(p))
            if a_log_lr_mult != 1.0 and 'mamba' in pname and 'A_log' in pname:
                a_log_params.append(p)
            elif alpha_lr_multiplier != 1.0 and pname == 'alpha':
                alpha_params.append(p)
            else:
                gate_params_list.append(p)
    for p in model.parameters():
        if p.requires_grad and id(p) not in gate_param_ids:
            lora_params.append(p)

    trainable_params = (sum(p.numel() for p in lora_params)
                        + sum(p.numel() for p in gate_params_list)
                        + sum(p.numel() for p in alpha_params))
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  LoRA params: {sum(p.numel() for p in lora_params):,}")
    print(f"  Gate params: {sum(p.numel() for p in gate_params_list):,}")
    print(f"  Trainable params: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.4f}%)")

    base_lr = config['training']['learning_rate']
    gate_lr_multiplier = config['training'].get('gate_lr_multiplier', 1.0)
    gate_lr_warmup_steps = config['training'].get('gate_lr_warmup_steps', 0)

    if gate_lr is not None:
        print(f"  Using separate gate LR: {gate_lr} (base: {base_lr})")
        optimizer = torch.optim.AdamW([
            {'params': lora_params, 'lr': base_lr},
            {'params': gate_params_list, 'lr': gate_lr},
        ], weight_decay=config['training']['weight_decay'])
        _has_two_groups = True
    elif gate_lr_multiplier != 1.0:
        print(f"  Gate LR multiplier: {gate_lr_multiplier}x (warmup {gate_lr_warmup_steps} steps)")
        optimizer = torch.optim.AdamW([
            {'params': lora_params, 'lr': base_lr},
            {'params': gate_params_list, 'lr': base_lr},  # multiplier applied via lambda below
        ], weight_decay=config['training']['weight_decay'])
        _has_two_groups = True
    else:
        optimizer = torch.optim.AdamW(
            lora_params + gate_params_list,
            lr=base_lr,
            weight_decay=config['training']['weight_decay'],
        )
        _has_two_groups = False

    # If α has its own LR multiplier, append it as a third group (on top of whatever
    # optimizer config above produced). Scheduler will apply the same base lambda to
    # all groups; the α group gets alpha_lr_multiplier baked into its initial lr.
    if alpha_params and alpha_lr_multiplier != 1.0:
        alpha_lr = base_lr * alpha_lr_multiplier
        print(f"  α-specific LR: {alpha_lr} ({alpha_lr_multiplier}x base, {len(alpha_params)} params)")
        optimizer.add_param_group({'params': alpha_params, 'lr': alpha_lr,
                                    'weight_decay': 0.0})

    # If mamba.A_log has its own LR multiplier, append it as a separate group.
    # A_log resists training because its gradient is small-mean / high-variance
    # (Adam dampens it) and dt_proj absorbs decay-scale gradient. Boosted LR with
    # zero weight_decay lets A_log actually move from log(1..d_state) init.
    if a_log_params and a_log_lr_mult != 1.0:
        a_log_lr = base_lr * a_log_lr_mult
        print(f"  A_log-specific LR: {a_log_lr} ({a_log_lr_mult}x base, {len(a_log_params)} params)")
        optimizer.add_param_group({'params': a_log_params, 'lr': a_log_lr,
                                    'weight_decay': 0.0})

    # LR scheduler with warmup
    warmup_steps = config['training'].get('warmup_steps', 0)
    total_steps = len(train_loader) * config['training']['num_epochs']
    if warmup_steps > 0:
        if gate_lr_multiplier != 1.0 and _has_two_groups:
            # Custom LambdaLR: both groups share warmup+decay; gate group also ramps its multiplier
            def _base_lambda(step):
                if step < warmup_steps:
                    return float(step) / float(max(1, warmup_steps))
                return max(0.0, float(total_steps - step) / float(max(1, total_steps - warmup_steps)))

            def _lora_lambda(step):
                return _base_lambda(step)

            def _gate_lambda(step):
                base = _base_lambda(step)
                if gate_lr_warmup_steps > 0 and step < gate_lr_warmup_steps:
                    mult = 1.0 + (gate_lr_multiplier - 1.0) * step / gate_lr_warmup_steps
                else:
                    mult = gate_lr_multiplier
                return base * mult

            # Build lambda list to match optimizer.param_groups length.
            # Extra groups (alpha, A_log) already bake their multiplier into initial_lr,
            # so they just need the standard warmup+decay lambda.
            lambdas = [_lora_lambda, _gate_lambda]
            for _ in optimizer.param_groups[2:]:  # alpha and/or A_log
                lambdas.append(_base_lambda)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambdas)
            print(f"  LR schedule: linear warmup ({warmup_steps}) + decay ({total_steps}); gate gets {gate_lr_multiplier}x over {gate_lr_warmup_steps} warmup steps")
        else:
            from transformers import get_linear_schedule_with_warmup
            scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
            print(f"  LR schedule: linear warmup ({warmup_steps} steps) + linear decay ({total_steps} total)")
    else:
        scheduler = None

    # Training loop
    print(f"\n{'='*50}")
    print(f"Starting Gated LoRA Training ({gate_type} gate)")
    print(f"{'='*50}")

    diag = DiagnosticsLogger(output_dir, log_every=config.get('logging', {}).get('log_every_n_steps', 50))
    diag.register_gate_hooks(model)
    diag.log_efficiency(model, optimizer, base_model_name=config['model']['base_model'], config_dict=config)
    if len(val_dataset) == 0:
        raise SystemExit(
            f"Validation split is empty ({len(full_dataset)} usable examples x "
            f"--val-split {args.val_split}). Either raise --max-train-samples or "
            f"--val-split. Note that examples whose prompt exceeds max_seq_length "
            f"({config['training']['max_seq_length']}) are dropped, so a small "
            f"max_seq_length can leave almost "
            f"nothing: MuSiQue needs 4096 and 2WikiMultihopQA 2048."
        )
    probe_batch = next(iter(val_loader))

    best_val_loss = float('inf')
    epoch_metrics = []
    torch.cuda.reset_peak_memory_stats()
    training_start = time.time()
    start_epoch = config['training'].get('start_epoch', 1)

    # Resume from checkpoint if specified
    resume_from = config['training'].get('resume_from', None)
    if resume_from:
        print(f"\nResuming from {resume_from}...")
        resume_ckpt = torch.load(resume_from, map_location='cpu', weights_only=False)
        for name, module in model.named_modules():
            if module.__class__.__name__ == 'MaLoRALinear':
                if name in resume_ckpt['gated_lora_states']:
                    states = resume_ckpt['gated_lora_states'][name]
                    module.lora_A.load_state_dict(states['lora_A'])
                    module.lora_B.load_state_dict(states['lora_B'])
                    module.gate.load_state_dict(states['gate'])
        best_val_loss = resume_ckpt.get('val_loss', float('inf'))
        print(f"  Resumed from epoch {resume_ckpt['epoch']}, val_loss={best_val_loss:.4f}")

    for epoch in range(start_epoch, config['training']['num_epochs'] + 1):
        print(f"\nEpoch {epoch}/{config['training']['num_epochs']}")

        # Train
        diag.start_epoch(epoch)
        epoch_start = time.time()
        train_loss, num_train_steps = train_epoch(
            model, train_loader, optimizer, config, epoch, scheduler, diag=diag,
            gate_params=gate_params_list, lora_params=lora_params, a_log_params=a_log_params,
            gate_modules=gate_modules,
        )
        epoch_train_time = time.time() - epoch_start
        print(f"  Train loss: {train_loss:.4f}")
        print(f"  Train time: {epoch_train_time:.1f}s ({epoch_train_time/num_train_steps:.2f}s/step)")

        # Validate
        model.eval()
        # Clear answer mask for validation (different seq lengths would cause shape mismatch)
        if config.get('gated_lora', {}).get('mask_answer_gates', False):
            set_answer_mask(model, None)
        val_loss = 0
        num_val_batches = 0
        val_start = time.time()

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                chunk_ends = batch.pop('chunk_ends', None)
                batch.pop('chunk_names', None)
                batch = {k: v.to(model.device) for k, v in batch.items()}
                if chunk_ends is not None:
                    from model.chunk_mamba_gate import (
                        set_chunk_ends_on_model,
                    )
                    set_chunk_ends_on_model(model, chunk_ends)
                try:
                    outputs = model(**batch)
                    val_loss += outputs.loss.item()
                    num_val_batches += 1
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    continue

        val_time = time.time() - val_start
        val_loss /= num_val_batches if num_val_batches > 0 else 1
        print(f"  Val loss: {val_loss:.4f}")
        print(f"  Val time: {val_time:.1f}s")

        diag.end_epoch(epoch, train_loss, val_loss, num_train_steps)
        try:
            diag.snapshot_gate_outputs(model, probe_batch, epoch, tokenizer=tokenizer)
        except Exception as e:
            print(f"  Gate snapshot failed: {e}")

        # GPU stats
        peak_mem_mb = torch.cuda.max_memory_allocated() / 1e6
        curr_mem_mb = torch.cuda.memory_allocated() / 1e6
        print(f"  GPU peak memory: {peak_mem_mb:.0f} MB | current: {curr_mem_mb:.0f} MB")

        epoch_metrics.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_time_s': round(epoch_train_time, 1),
            'val_time_s': round(val_time, 1),
            'time_per_step_s': round(epoch_train_time / num_train_steps, 3),
            'num_train_steps': num_train_steps,
            'peak_gpu_memory_mb': round(peak_mem_mb, 0),
        })

        # Save checkpoint
        gated_lora_states = {}
        for name, module in model.named_modules():
            if module.__class__.__name__ == 'MaLoRALinear':
                gated_lora_states[name] = {
                    'lora_A': module.lora_A.state_dict(),
                    'lora_B': module.lora_B.state_dict(),
                    'gate': module.gate.state_dict(),
                }

        checkpoint = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'config': gated_lora_config,
            'gated_lora_states': gated_lora_states,
            'gate_modules': [g.state_dict() for g in gate_modules],
        }

        checkpoint_path = output_dir / f"gated_lora_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f"  Saved checkpoint to {checkpoint_path}")

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = output_dir / "gated_lora_best.pt"
            torch.save(checkpoint, best_path)
            print(f"  New best model! (val_loss: {val_loss:.4f})")

        # Auto gate analysis after each epoch
        try:
            gate_n_samples = config.get('logging', {}).get('gate_analysis_samples', 100)
            run_gate_analysis_on_eval(model, tokenizer, dataset_name, config, n_samples=gate_n_samples)
        except Exception as e:
            print(f"  Gate analysis failed: {e}")

        # Auto eval after each epoch
        try:
            if dataset_name == 'quality':
                print(f"\n  Running QuALITY eval on checkpoint epoch {epoch}...")
                from datasets import load_dataset as _load_dataset
                from data.quality import format_quality_prompt, format_quality_prompt_qfirst
                qf = config.get('dataset', {}).get('question_first', False)
                fmt_fn = format_quality_prompt_qfirst if qf else format_quality_prompt
                eval_ds = _load_dataset('emozilla/quality', split='validation')

                option_tokens = {}
                for letter in ['A', 'B', 'C', 'D']:
                    token_id = tokenizer.encode(letter, add_special_tokens=False)
                    option_tokens[letter] = token_id[0]

                correct = 0
                correct_hard = 0
                total_eval = 0
                total_hard = 0
                max_length = config.get('evaluation', {}).get('max_length', 16384)

                with torch.no_grad():
                    for idx, item in enumerate(eval_ds):
                        prompt = fmt_fn(item['article'], item['question'],
                                       item['options'], answer_idx=None)
                        inputs = tokenizer(prompt, return_tensors='pt', truncation=True,
                                          max_length=max_length)
                        inputs = {k: v.to(model.device) for k, v in inputs.items()}
                        try:
                            outputs = model(**inputs)
                            logits = outputs.logits[0, -1, :]
                            option_logits = {l: logits[t].item() for l, t in option_tokens.items()}
                            predicted = max(option_logits, key=option_logits.get)
                            predicted_idx = ord(predicted) - ord('A')
                            if predicted_idx == item['answer']:
                                correct += 1
                            total_eval += 1
                            if item['hard']:
                                total_hard += 1
                                if predicted_idx == item['answer']:
                                    correct_hard += 1
                        except torch.cuda.OutOfMemoryError:
                            torch.cuda.empty_cache()
                            continue

                        if (idx + 1) % 500 == 0:
                            print(f"    Eval progress: {idx+1}/{len(eval_ds)} "
                                  f"(acc so far: {100*correct/total_eval:.1f}%)")

                print(f"  Epoch {epoch} QuALITY Eval:")
                print(f"    Overall Accuracy: {100*correct/total_eval:.1f}% ({correct}/{total_eval})")
                if total_hard > 0:
                    print(f"    Hard-only Accuracy: {100*correct_hard/total_hard:.1f}% ({correct_hard}/{total_hard})")

            elif dataset_name == 'hotpotqa' and not config.get('skip_intraining_eval', False):
                print(f"\n  Running HotpotQA eval on checkpoint epoch {epoch}...")
                from datasets import load_dataset as _load_dataset
                eval_ds = _load_dataset('hotpot_qa', 'fullwiki', split='validation')
                n_eval = min(500, len(eval_ds))
                correct_em = 0
                total_eval = 0
                eval_cfg = config.get('evaluation', {})
                max_new_tokens = eval_cfg.get('max_new_tokens', 32)

                with torch.no_grad():
                    for idx in range(n_eval):
                        item = eval_ds[idx]
                        context = ' '.join(item['context']['sentences'][0]) if item['context']['sentences'] else ''
                        prompt = f"Context: {context}\n\nQuestion: {item['question']}\n\nAnswer:"
                        inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=2048)
                        inputs = {k: v.to(model.device) for k, v in inputs.items()}
                        try:
                            gen = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                                 do_sample=False, temperature=None, top_p=None)
                            pred = tokenizer.decode(gen[0][inputs['input_ids'].shape[1]:],
                                                    skip_special_tokens=True).strip().lower()
                            gold = item['answer'].strip().lower()
                            if pred == gold:
                                correct_em += 1
                            total_eval += 1
                        except torch.cuda.OutOfMemoryError:
                            torch.cuda.empty_cache()
                            continue

                        if (idx + 1) % 100 == 0:
                            print(f"    Eval progress: {idx+1}/{n_eval} "
                                  f"(EM so far: {100*correct_em/total_eval:.1f}%)")

                print(f"  Epoch {epoch} HotpotQA Eval ({total_eval} samples):")
                print(f"    Exact Match: {100*correct_em/total_eval:.1f}% ({correct_em}/{total_eval})")
        except Exception as e:
            print(f"  Auto-eval failed: {e}")

        model.train()  # back to training mode

    total_training_time = time.time() - training_start

    # Save metrics JSON
    mode_tag = "direct" if direct_mode else ("hproj" if hproj_mode else "bn")
    output_tag = "scalar" if scalar_output else "diag"
    metrics = {
        'method': f'malora_{gate_type}_{mode_tag}_{output_tag}',
        'model': config['model']['base_model'],
        'dataset': dataset_name,
        'rank': rank,
        'gate_type': gate_type,
        'mode': mode_tag,
        'output_type': output_tag,
        'target_modules': gated_lora_config['target_modules'],
        'trainable_params': trainable_params,
        'total_params': total_params,
        'trainable_pct': round(100 * trainable_params / total_params, 4),
        'total_training_time_s': round(total_training_time, 1),
        'avg_time_per_step_s': round(sum(e['time_per_step_s'] for e in epoch_metrics) / len(epoch_metrics), 3),
        'peak_gpu_memory_mb': round(torch.cuda.max_memory_allocated() / 1e6, 0),
        'best_val_loss': best_val_loss,
        'epochs': epoch_metrics,
    }
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    diag.finish()

    print(f"\n{'='*50}")
    print("Training Complete!")
    print(f"{'='*50}")
    print(f"\nCheckpoints saved to: {output_dir}")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Total training time: {total_training_time:.1f}s ({total_training_time/60:.1f}min)")
    print(f"Peak GPU memory: {torch.cuda.max_memory_allocated()/1e6:.0f} MB")
    print(f"Metrics saved to: {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
