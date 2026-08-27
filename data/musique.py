"""
MuSiQue dataset loader for training.

Dataset: dgslibisey/MuSiQue
Format: Multi-hop QA (3-4 hops) with multiple context paragraphs

When `lora_pool_path` is set, loads pre-computed canonical 10k pool from JSON
(e.g., data/preprocessed_10k/musique_canonical_n10000.json) instead of doing
HF load + filter + stratification. The canonical pool guarantees identical
samples across tokenizers and chunk-detect compatibility.
"""

import json
from pathlib import Path

import torch
from datasets import load_dataset
from torch.utils.data import Dataset


def format_musique_prompt(question, paragraphs, answer=None):
    """
    Format MuSiQue example as prompt.

    Args:
        question: The question
        paragraphs: List of dicts with 'title' and 'paragraph_text'
        answer: Answer text (optional, for training)

    Returns:
        Formatted prompt
    """
    context_parts = []
    for para in paragraphs:
        context_parts.append(f"{para['title']}: {para['paragraph_text']}")

    context = '\n\n'.join(context_parts)

    if answer is not None:
        return f"""Context:
{context}

Question: {question}

Answer: {answer}"""
    else:
        return f"""Context:
{context}

Question: {question}

Answer:"""


class MuSiQueDataset(Dataset):
    """
    MuSiQue dataset for supervised fine-tuning.

    Multi-hop reasoning (3-4 hops) over multiple paragraphs.
    """

    def __init__(self, tokenizer=None, max_length=2048, split='train',
                 max_samples=None, with_chunks=False, lora_pool_path=None):
        """
        Args:
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length (total)
            split: 'train' or 'validation'
            max_samples: Limit number of samples (for quick testing)
            with_chunks: if True, route __getitem__ through _getitem_with_chunks
                         which detects per-paragraph chunk boundaries via
                         offset_mapping and applies AMASK on the A chunk so
                         the answer body doesn't leak into chunk pooling.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.with_chunks = with_chunks

        # Canonical pool path: load from JSON instead of HF + filter
        if lora_pool_path is not None and split == 'train':
            print(f"Loading MuSiQue from canonical pool {lora_pool_path}")
            with open(lora_pool_path) as _f:
                _pool = json.load(_f)
            self.examples = []
            for it in _pool['examples']:
                full_text = format_musique_prompt(it['question'], it['paragraphs'], it['answer'])
                self.examples.append({
                    'question': it['question'],
                    'answer': it['answer'],
                    'answer_aliases': it.get('answer_aliases', []),
                    'paragraphs': it['paragraphs'],
                    'full_text': full_text,
                    'num_hops': it.get('num_hops', 0),
                })
            print(f"  Loaded {len(self.examples)} canonical examples "
                  f"(filter_rule={_pool.get('filter_rule', 'unknown')})")
            return

        print(f"Loading MuSiQue {split} split...")
        dataset = load_dataset('dgslibisey/MuSiQue', split=split)

        # Group examples by hop count for stratified sampling
        from collections import defaultdict
        import random
        hop_buckets = defaultdict(list)
        skipped_no_answer = 0
        skipped_unanswerable = 0
        skipped_too_long = 0

        for item in dataset:
            # Skip unanswerable questions
            if not item.get('answerable', True):
                skipped_unanswerable += 1
                continue

            question = item['question']
            answer = item['answer']

            if not answer or answer.strip() == '':
                skipped_no_answer += 1
                continue

            paragraphs = item['paragraphs']

            # Create formatted prompt
            full_text = format_musique_prompt(question, paragraphs, answer)

            # Check actual token length — skip if exceeds max_length (no truncation)
            token_len = len(self.tokenizer.encode(full_text, add_special_tokens=True))
            if token_len > max_length:
                skipped_too_long += 1
                continue

            num_hops = len(item.get('question_decomposition', []))
            hop_buckets[num_hops].append({
                'question': question,
                'answer': answer,
                'answer_aliases': item.get('answer_aliases', []),
                'paragraphs': paragraphs,
                'full_text': full_text,
                'num_hops': num_hops,
            })

        # Stratified sampling: proportional to available data per hop count
        self.examples = []
        if max_samples is not None:
            total_available = sum(len(v) for v in hop_buckets.values())
            rng = random.Random(42)
            for hop_count in sorted(hop_buckets.keys()):
                bucket = hop_buckets[hop_count]
                rng.shuffle(bucket)
                # Proportional allocation
                n_select = int(max_samples * len(bucket) / total_available)
                n_select = max(1, min(n_select, len(bucket)))
                self.examples.extend(bucket[:n_select])
            # Fill remaining if rounding left us short
            rng.shuffle(self.examples)
        else:
            for bucket in hop_buckets.values():
                self.examples.extend(bucket)
            random.Random(42).shuffle(self.examples)

        # Print hop distribution
        from collections import Counter
        hop_dist = Counter(ex.get('num_hops', 0) for ex in self.examples)
        print(f"  Loaded {len(self.examples)} examples")
        print(f"  Hop distribution: {dict(sorted(hop_dist.items()))}")
        if skipped_unanswerable > 0:
            print(f"  Skipped {skipped_unanswerable} unanswerable examples")
        if skipped_no_answer > 0:
            print(f"  Skipped {skipped_no_answer} examples with no answer")
        if skipped_too_long > 0:
            print(f"  Skipped {skipped_too_long} examples exceeding {max_length} tokens")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        if self.with_chunks:
            return self._getitem_with_chunks(idx)
        example = self.examples[idx]

        # Tokenize question + "Answer: " + answer (the part we MUST keep)
        question_text = f"""Question: {example['question']}

Answer: {example['answer']}"""

        question_encoded = self.tokenizer(
            question_text,
            add_special_tokens=False,
            padding=False,
        )
        question_tokens = len(question_encoded['input_ids'])

        # Reserve space for question+answer (add buffer for special tokens)
        reserved_tokens = question_tokens + 10
        available_for_context = self.max_length - reserved_tokens

        if available_for_context < 100:
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

            labels = input_ids.clone()
            labels[:-20] = -100
            labels[labels == self.tokenizer.pad_token_id] = -100

        else:
            # Build context from paragraphs
            context_parts = []
            for para in example['paragraphs']:
                context_parts.append(f"{para['title']}: {para['paragraph_text']}")
            context_str = '\n\n'.join(context_parts)

            # Truncate context to fit
            context_encoded = self.tokenizer(
                f"Context:\n{context_str}",
                add_special_tokens=False,
                truncation=True,
                max_length=available_for_context,
                padding=False,
            )

            eos = [self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id is not None else []
            combined_ids = context_encoded['input_ids'] + question_encoded['input_ids'] + eos

            if self.tokenizer.bos_token_id is not None:
                combined_ids = [self.tokenizer.bos_token_id] + combined_ids

            input_ids = torch.tensor(combined_ids, dtype=torch.long)
            attention_mask = torch.ones_like(input_ids)

            if len(input_ids) > self.max_length:
                input_ids = input_ids[:self.max_length]
                attention_mask = attention_mask[:self.max_length]

            # Create labels: mask context+question, keep only answer + EOS
            labels = input_ids.clone()

            answer_tokens = self.tokenizer.encode(example['answer'], add_special_tokens=False)
            answer_plus_eos = len(answer_tokens) + len(eos)
            answer_start = len(input_ids) - answer_plus_eos

            labels[:answer_start] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }


    def _getitem_with_chunks(self, idx):
        """Tokenize full_text with offset_mapping, detect per-paragraph chunks.

        Same prompt structure as HotpotQA (Context: ... Question: ... Answer:),
        so we reuse detect_hotpotqa_chunks. Applies AMASK on the A chunk: the
        chunk's end is set to the position right after "Answer:" (no answer
        body inside the chunk), eliminating the answer-token leak through
        chunk_rep pooling.
        """
        import re
        from chunking.chunk_detector import detect_hotpotqa_chunks

        example = self.examples[idx]
        text = example['full_text']

        assert getattr(self.tokenizer, 'is_fast', False), \
            "with_chunks=True requires a fast tokenizer (offset_mapping)"

        enc = self.tokenizer(
            text,
            return_tensors='pt',
            return_offsets_mapping=True,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
        )
        input_ids = enc['input_ids'][0]
        attention_mask = enc['attention_mask'][0]
        offsets = enc['offset_mapping'][0].tolist()

        # AMASK: A chunk truncated to colon (no answer body in chunk pool).
        answer_marker = re.search(r"\nAnswer:", text)
        if answer_marker is None:
            raise RuntimeError(f"No 'Answer:' in text for idx={idx}")
        answer_end_char = answer_marker.end()
        chunks = detect_hotpotqa_chunks(text, offsets, answer_end_char=answer_end_char)
        chunk_ends = [c['end_tok'] for c in chunks]
        chunk_names = [c['name'] for c in chunks]

        # Build labels: mask everything except answer region
        answer_match = re.search(r"\nAnswer:\s+", text)
        if answer_match is None:
            raise RuntimeError(f"No 'Answer: <body>' in text for idx={idx}")
        answer_start_char = answer_match.end()
        answer_start_tok = None
        for i, (s, _) in enumerate(offsets):
            if s >= answer_start_char:
                answer_start_tok = i
                break
        if answer_start_tok is None:
            answer_start_tok = len(input_ids) - 1
        labels = torch.full_like(input_ids, -100)
        labels[answer_start_tok:] = input_ids[answer_start_tok:]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'chunk_ends': chunk_ends,
            'chunk_names': chunk_names,
        }


if __name__ == "__main__":
    from transformers import AutoTokenizer

    print("Testing MuSiQue data loader...")

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    tokenizer.pad_token = tokenizer.eos_token

    dataset = MuSiQueDataset(
        tokenizer=tokenizer,
        max_length=2048,
        split='train',
        max_samples=5,
    )

    example = dataset[0]
    print(f"\nFirst example:")
    print(f"  Input IDs shape: {example['input_ids'].shape}")
    print(f"  Question: {dataset.examples[0]['question']}")
    print(f"  Answer: {dataset.examples[0]['answer']}")
    print(f"  Paragraphs: {len(dataset.examples[0]['paragraphs'])}")

    n_masked = (example['labels'] == -100).sum().item()
    n_total = len(example['labels'])
    n_valid = n_total - n_masked
    print(f"  Valid label tokens: {n_valid}/{n_total} ({100*n_valid/n_total:.1f}%)")
