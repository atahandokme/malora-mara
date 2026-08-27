"""
Commonsense Reasoning dataset loader for training.

Dataset: commonsense_170k from LLM-Adapters (AGI-Edgerunners)
Combined training data from 8 commonsense reasoning benchmarks:
  BoolQ, PIQA, SIQA, HellaSwag, WinoGrande, ARC-e, ARC-c, OBQA

Format: instruction-following with instruction + output
"""

import json
import torch
from torch.utils.data import Dataset


class CommonsenseDataset(Dataset):
    """Commonsense reasoning dataset for supervised fine-tuning."""

    def __init__(self, tokenizer, max_length=512, data_path='data/commonsense_170k.json',
                 max_samples=None, seed=42):
        with open(data_path) as f:
            data = json.load(f)

        if max_samples:
            # Seed-deterministic random sample (not first-N) for fair subsetting.
            import random
            rng = random.Random(seed)
            rng.shuffle(data)
            data = data[:max_samples]

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        skipped = 0
        for item in data:
            instruction = item['instruction'].strip()
            inp = item.get('input', '').strip()
            output = item.get('output', '').strip()

            if not instruction or not output:
                skipped += 1
                continue

            # Build prompt in Alpaca format (used by LLM-Adapters)
            if inp:
                prompt = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{inp}\n\n### Response:\n"
            else:
                prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"

            full_text = prompt + output

            self.examples.append({
                'prompt': prompt,
                'output': output,
                'full_text': full_text,
            })

        print(f"  Loaded {len(self.examples)} commonsense training examples")
        if skipped:
            print(f"  Skipped {skipped} empty examples")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        # Tokenize full text (prompt + output)
        full_encoded = self.tokenizer(
            example['full_text'],
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors='pt',
        )
        input_ids = full_encoded['input_ids'].squeeze(0)
        attention_mask = full_encoded['attention_mask'].squeeze(0)

        # Create labels: mask prompt tokens, keep only output tokens
        prompt_encoded = self.tokenizer(
            example['prompt'],
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        prompt_len = len(prompt_encoded['input_ids'])

        labels = input_ids.clone()
        labels[:prompt_len] = -100
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }


if __name__ == "__main__":
    from transformers import AutoTokenizer

    print("Testing Commonsense data loader...")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = CommonsenseDataset(tokenizer=tokenizer, max_length=512, max_samples=100)
    print(f"Dataset size: {len(dataset)}")

    sample = dataset[0]
    print(f"input_ids shape: {sample['input_ids'].shape}")
    print(f"Prompt: {dataset.examples[0]['prompt'][:200]}...")
    print(f"Output: {dataset.examples[0]['output'][:100]}")

    # Check label masking
    n_masked = (sample['labels'] == -100).sum().item()
    n_total = len(sample['labels'])
    print(f"Labels: {n_masked} masked / {n_total} total ({100*n_masked/n_total:.0f}% prompt)")
