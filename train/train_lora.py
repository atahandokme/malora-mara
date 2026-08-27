"""
Training script for LoRA / AdaLoRA baselines.

Fair comparison baselines for Gated LoRA: same model, data, hyperparameters.
Uses PEFT library. Supports both HotpotQA and MuSiQue datasets.

Usage:
  # LoRA on MuSiQue:
  python train_lora.py --config configs/musique_lora.yaml

  # AdaLoRA on MuSiQue:
  python train_lora.py --config configs/musique_adalora.yaml
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
from peft import LoraConfig, get_peft_model, TaskType
from diagnostics import DiagnosticsLogger

from data.musique import MuSiQueDataset
from data.twowikimultihopqa import TwoWikiMultihopQADataset
from data.commonsense import CommonsenseDataset
from data.gsm8k import GSM8KDataset


def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train_epoch(model, dataloader, optimizer, config, epoch, scheduler=None, diag=None):
    """Train for one epoch."""
    model.train()

    total_loss = 0
    num_batches = 0

    grad_accum_steps = config['training']['gradient_accumulation_steps']

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    skipped_oom = 0

    for batch_idx, batch in enumerate(pbar):
        batch = {k: v.to(model.device) for k, v in batch.items()}

        try:
            outputs = model(**batch)
            loss = outputs.loss / grad_accum_steps

            if batch_idx == 0:
                valid_labels = batch['labels'][batch['labels'] != -100]
                print(f"\n  First batch stats:")
                print(f"    Loss: {loss.item() * grad_accum_steps:.4f}")
                print(f"    Sequence length: {batch['input_ids'].shape[1]}")
                print(f"    Valid labels: {len(valid_labels)} tokens")
                print(f"    Logits range: [{outputs.logits.min().item():.2f}, {outputs.logits.max().item():.2f}]")

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n  NaN/Inf detected at batch {batch_idx}, skipping...")
                continue

            loss.backward()

            if diag is not None:
                ntok = int(batch['attention_mask'].sum().item())
                diag.log_step(batch_idx, loss.item() * grad_accum_steps, model, optimizer, batch_tokens=ntok)

            if (batch_idx + 1) % grad_accum_steps == 0:
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
    parser.add_argument('--config', type=str, default='configs/lora.yaml')
    parser.add_argument('--val-split', type=float, default=0.05)
    parser.add_argument('--max-train-samples', type=int, default=None)
    parser.add_argument('--lora-pool-path', type=str, default=None,
                        help='Path to canonical 10k JSON; if set, dataset loads from this instead of HF + filter')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed for data split + LoRA init + shuffle. '
                             'If != 42, embedded in output dir name as _seedNN.')
    parser.add_argument('--output-root', type=str, default=None,
                        help='Override output directory root (e.g., outputs/canonical_pool)')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    print(f"Loaded config from {args.config}")

    # Create output directory with descriptive name
    dataset_name = config.get('dataset', {}).get('name', 'hotpotqa')
    method = config.get('lora', {}).get('method', config.get('method', 'lora'))
    rank = config.get('lora', {}).get('rank', 16)
    use_dora = config['lora'].get('use_dora', False)
    if use_dora:
        method = 'dora'
    # Short model name: "Qwen/Qwen2.5-7B-Instruct" -> "qwen2.5-7b"
    base_model = config['model']['base_model']
    model_short = base_model.split('/')[-1].lower().replace('-instruct', '').replace('-it', '')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    seed_tag = "" if args.seed == 42 else f"_seed{args.seed}"
    output_root = Path(args.output_root or config.get('output_root', 'outputs'))
    output_dir = output_root / f"{dataset_name}_{method}_r{rank}_{model_short}{seed_tag}_{timestamp}"
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
        print("\nLoading MuSiQue dataset...")
        full_dataset = MuSiQueDataset(
            tokenizer=tokenizer,
            max_length=config['training']['max_seq_length'],
            split='train',
            max_samples=max_samples,
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
    else:
        print("\nLoading HotpotQA dataset...")
        from data.hotpotqa import HotpotQADataset
        full_dataset = HotpotQADataset(
            tokenizer=tokenizer,
            max_length=config['training']['max_seq_length'],
            split='train',
            max_samples=max_samples,
            lora_pool_path=lora_pool_path,
        )

    # Split into train/val. Seed controls split + later shuffling.
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    # Seed torch/numpy/python so LoRA init + dropout differ across seeds
    import random as _py_random, numpy as _np
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    _np.random.seed(args.seed)
    _py_random.seed(args.seed)

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")

    # Collate function for dynamic padding (same as SRA training)
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

        return {
            'input_ids': torch.stack(input_ids),
            'attention_mask': torch.stack(attention_mask),
            'labels': torch.stack(labels),
        }

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

    model = AutoModelForCausalLM.from_pretrained(
        config['model']['base_model'],
        torch_dtype=torch_dtype,
        device_map=config['hardware']['device'],
        attn_implementation="sdpa",
    )

    # Apply LoRA or AdaLoRA
    if method == 'adalora':
        from peft import AdaLoraConfig
        print("\nApplying AdaLoRA...")
        peft_config = AdaLoraConfig(
            task_type=TaskType.CAUSAL_LM,
            init_r=config['lora']['init_r'],
            target_r=config['lora']['rank'],
            lora_alpha=config['lora']['alpha'],
            lora_dropout=config['lora']['dropout'],
            target_modules=config['lora']['target_modules'],
            bias="none",
            total_step=config['lora'].get('total_step'),
            tinit=config['lora'].get('tinit', 200),
            tfinal=config['lora'].get('tfinal', 1000),
            deltaT=config['lora'].get('deltaT', 10),
        )
    else:
        use_dora = config['lora'].get('use_dora', False)
        print(f"\nApplying {'DoRA' if use_dora else 'LoRA'}...")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config['lora']['rank'],
            lora_alpha=config['lora']['alpha'],
            lora_dropout=config['lora']['dropout'],
            target_modules=config['lora']['target_modules'],
            bias="none",
            use_dora=use_dora,
            init_lora_weights=config['lora'].get('init_lora_weights', True),
        )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # PiSSA: save the initial (pre-training) adapter so epoch checkpoints can be
    # converted back to standard LoRA deltas applicable to the original base.
    _ilw = config['lora'].get('init_lora_weights', True)
    pissa_init_dir = None
    if isinstance(_ilw, str) and _ilw.lower().startswith('pissa'):
        pissa_init_dir = output_dir / "pissa_init"
        model.save_pretrained(pissa_init_dir)
        print(f"  Saved PiSSA initial adapter to {pissa_init_dir} (residual conversion ref)")

    # Enable gradient checkpointing
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    print(f"  Enabled gradient checkpointing")

    # Setup optimizer (only LoRA params are trainable)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
    )

    # LR scheduler with warmup
    warmup_steps = config['training'].get('warmup_steps', 0)
    total_steps = len(train_loader) * config['training']['num_epochs']
    if warmup_steps > 0:
        from transformers import get_linear_schedule_with_warmup
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        print(f"  LR schedule: linear warmup ({warmup_steps} steps) + linear decay ({total_steps} total)")
    else:
        scheduler = None

    # Count params
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.4f}%)")

    # Training loop
    print(f"\n{'='*50}")
    print(f"Starting {method.upper()} Training on {dataset_name}")
    print(f"{'='*50}")

    diag = DiagnosticsLogger(output_dir, log_every=config.get('logging', {}).get('log_every_n_steps', 50))
    diag.log_efficiency(model, optimizer, base_model_name=config['model']['base_model'], config_dict=config)
    probe_batch = next(iter(val_loader))

    best_val_loss = float('inf')
    epoch_metrics = []
    torch.cuda.reset_peak_memory_stats()
    training_start = time.time()

    for epoch in range(1, config['training']['num_epochs'] + 1):
        print(f"\nEpoch {epoch}/{config['training']['num_epochs']}")

        # Train
        diag.start_epoch(epoch)
        epoch_start = time.time()
        train_loss, num_train_steps = train_epoch(model, train_loader, optimizer, config, epoch, scheduler, diag=diag)
        epoch_train_time = time.time() - epoch_start
        print(f"  Train loss: {train_loss:.4f}")
        print(f"  Train time: {epoch_train_time:.1f}s ({epoch_train_time/num_train_steps:.2f}s/step)")

        # Validate
        model.eval()
        val_loss = 0
        num_val_batches = 0
        val_start = time.time()

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                batch = {k: v.to(model.device) for k, v in batch.items()}
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

        # Save LoRA checkpoint (PiSSA: convert to standard-LoRA delta vs original base)
        checkpoint_path = output_dir / f"lora_epoch_{epoch}"
        if pissa_init_dir is not None:
            model.save_pretrained(checkpoint_path, path_initial_model_for_weight_conversion=str(pissa_init_dir))
        else:
            model.save_pretrained(checkpoint_path)
        print(f"  Saved LoRA checkpoint to {checkpoint_path}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = output_dir / "lora_best"
            if pissa_init_dir is not None:
                model.save_pretrained(best_path, path_initial_model_for_weight_conversion=str(pissa_init_dir))
            else:
                model.save_pretrained(best_path)
            print(f"  New best model! (val_loss: {val_loss:.4f})")

        # Save training state for comparison
        torch.save({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, output_dir / f"training_state_epoch_{epoch}.pt")

    total_training_time = time.time() - training_start

    # Save metrics JSON
    metrics = {
        'method': method,
        'model': config['model']['base_model'],
        'dataset': dataset_name,
        'rank': rank,
        'target_modules': config['lora']['target_modules'],
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
