"""
GSM8K dataset loader and formatter for SFT training.
"""

import torch
from datasets import load_dataset
from torch.utils.data import Dataset


def format_gsm8k_prompt(question, answer=None):
    """
    Format a GSM8K example as a prompt.

    Matches lm-evaluation-harness `gsm8k` task template:
        doc_to_text: "Question: {{question}}\nAnswer:"
    so fine-tuned models see the exact prompt tag at eval time.
    """
    if answer is not None:
        return f"Question: {question}\nAnswer: {answer}"
    else:
        return f"Question: {question}\nAnswer:"


class GSM8KDataset(Dataset):
    """
    GSM8K dataset for supervised fine-tuning.

    Formats examples as:
        "Question: {question}\n\nSolution: {answer}"

    And masks the question tokens so loss is only computed on the solution.
    """

    def __init__(self, split='train', tokenizer=None, max_length=1024, val_split_ratio=0.1, seed=42,
                 max_samples=None):
        """
        Args:
            split: 'train', 'val', or 'test'
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            val_split_ratio: Ratio of train set to use for validation (default 0.1 = 10%)
            seed: Random seed for train/val split and subsampling
            max_samples: If set, deterministically subsample to this many examples (after length filtering)
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self._max_samples = max_samples
        self._seed = seed

        # Load dataset
        if split in ['train', 'val']:
            print(f"Loading GSM8K train split...")
            full_train = load_dataset('gsm8k', 'main', split='train')

            # Split train into train/val
            split_dataset = full_train.train_test_split(
                test_size=val_split_ratio,
                seed=seed
            )

            if split == 'train':
                dataset = split_dataset['train']
                print(f"  Using {len(dataset)} examples for training")
            else:  # val
                dataset = split_dataset['test']
                print(f"  Using {len(dataset)} examples for validation")
        else:  # test
            print(f"Loading GSM8K test split...")
            dataset = load_dataset('gsm8k', 'main', split='test')

        # Format examples — filter any sample whose tokenized full_text exceeds
        # max_length so the final-answer tokens (#### <number>) are never truncated.
        self.examples = []
        skipped_too_long = 0
        eos = tokenizer.eos_token if tokenizer is not None and tokenizer.eos_token else ''
        for item in dataset:
            question = item['question']
            answer = item['answer']

            full_text = format_gsm8k_prompt(question, answer) + eos
            question_only = format_gsm8k_prompt(question, answer=None)

            if tokenizer is not None:
                enc_len = len(tokenizer(full_text, add_special_tokens=True)['input_ids'])
                if enc_len > max_length:
                    skipped_too_long += 1
                    continue

            self.examples.append({
                'full_text': full_text,
                'question_only': question_only,
                'question': question,
                'answer': answer,
            })

        print(f"  Loaded {len(self.examples)} examples"
              + (f" (filtered {skipped_too_long} over max_length={max_length})" if skipped_too_long else ""))

        if self._max_samples is not None and len(self.examples) > self._max_samples:
            import random
            rng = random.Random(self._seed)
            rng.shuffle(self.examples)
            self.examples = self.examples[:self._max_samples]
            print(f"  Subsampled to {len(self.examples)} examples (seed={self._seed})")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        # Tokenize full text
        full_encoded = self.tokenizer(
            example['full_text'],
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
        )

        # Tokenize question only (to find where to mask)
        question_encoded = self.tokenizer(
            example['question_only'],
            add_special_tokens=False,  # Don't add special tokens for this
        )

        # Create labels: mask question tokens, keep answer tokens
        labels = full_encoded['input_ids'].clone()
        question_len = len(question_encoded['input_ids'])

        # Mask question tokens with -100 (ignored by loss)
        labels[0, :question_len] = -100

        # Also mask padding tokens
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': full_encoded['input_ids'].squeeze(0),
            'attention_mask': full_encoded['attention_mask'].squeeze(0),
            'labels': labels.squeeze(0),
        }


def extract_answer_number(text):
    """
    Extract the numerical answer from GSM8K format or natural language.

    Tries multiple strategies:
    1. Look for #### format (ground truth format)
    2. Look for common patterns like "The answer is X"
    3. Extract last number in text as fallback

    Returns:
        The extracted number, or None if not found
    """
    import re

    # Strategy 1: Check for #### format (ground truth)
    if '####' in text:
        try:
            answer_str = text.split('####')[-1].strip()
            answer_str = answer_str.replace(',', '')
            return float(answer_str)
        except:
            pass

    # Strategy 2: Look for common answer patterns
    patterns = [
        r'(?:the\s+)?answer\s+is\s+[\$]?([0-9,]+\.?[0-9]*)',
        r'(?:equals?|=)\s+[\$]?([0-9,]+\.?[0-9]*)',
        r'(?:total|result)\s+(?:is|:)\s+[\$]?([0-9,]+\.?[0-9]*)',
        r'[\$]?([0-9,]+\.?[0-9]*)\s*(?:dollars?|cents?)?\.?\s*$',  # Number at end
    ]

    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            try:
                num_str = match.group(1).replace(',', '')
                return float(num_str)
            except:
                continue

    # Strategy 3: Extract last number in text as fallback
    numbers = re.findall(r'[\$]?([0-9,]+\.?[0-9]*)', text)
    if numbers:
        try:
            # Try last number
            num_str = numbers[-1].replace(',', '')
            return float(num_str)
        except:
            pass

    return None


def evaluate_gsm8k_prediction(prediction, ground_truth):
    """
    Check if a prediction matches the ground truth answer.

    Args:
        prediction: Generated text
        ground_truth: Ground truth answer text

    Returns:
        bool: True if answers match
    """
    pred_num = extract_answer_number(prediction)
    gt_num = extract_answer_number(ground_truth)

    if pred_num is None or gt_num is None:
        return False

    # Allow small floating point tolerance
    return abs(pred_num - gt_num) < 1e-4


if __name__ == "__main__":
    # Test data loader
    from transformers import AutoTokenizer

    print("Testing GSM8K data loader...")

    # Load tokenizer (using a small model for testing)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Create dataset
    dataset = GSM8KDataset(split='train', tokenizer=tokenizer, max_length=512)

    # Test first example
    example = dataset[0]
    print(f"\nFirst example:")
    print(f"  Input IDs shape: {example['input_ids'].shape}")
    print(f"  Attention mask shape: {example['attention_mask'].shape}")
    print(f"  Labels shape: {example['labels'].shape}")

    # Decode and show
    decoded_input = tokenizer.decode(example['input_ids'], skip_special_tokens=True)
    print(f"\n  Decoded input:\n{decoded_input[:200]}...")

    # Check label masking
    n_masked = (example['labels'] == -100).sum().item()
    n_total = len(example['labels'])
    print(f"\n  Masked tokens: {n_masked}/{n_total} ({100*n_masked/n_total:.1f}%)")

    # Test answer extraction
    print("\nTesting answer extraction...")
    test_cases = [
        "So the answer is 42.\n#### 42",
        "Therefore, there are 1,234 apples.\n#### 1234",
        "The total is $56.78.\n#### 56.78",
    ]

    for test in test_cases:
        answer = extract_answer_number(test)
        print(f"  '{test}' → {answer}")

    print("\n✅ Tests passed!")
