"""
HotpotQA dataset loader for training.

Dataset: hotpot_qa/distractor
Format: Multi-hop QA with multiple context paragraphs
"""

import torch
from datasets import load_dataset
from torch.utils.data import Dataset


def format_hotpotqa_prompt(question, context_paragraphs, answer=None):
    """
    Format HotpotQA example as prompt.

    Args:
        question: The question
        context_paragraphs: List of (title, sentences) tuples
        answer: Answer text (optional, for training)

    Returns:
        Formatted prompt
    """
    # Build context from paragraphs
    context_parts = []
    for title, sentences in context_paragraphs:
        # Join sentences into paragraph
        paragraph = ' '.join(sentences)
        context_parts.append(f"{title}: {paragraph}")

    context = '\n\n'.join(context_parts)

    if answer is not None:
        # Training format
        return f"""Context:
{context}

Question: {question}

Answer: {answer}"""
    else:
        # Inference format
        return f"""Context:
{context}

Question: {question}

Answer:"""


class HotpotQADataset(Dataset):
    """
    HotpotQA dataset for supervised fine-tuning.

    Multi-hop reasoning over multiple paragraphs with controllable context length.
    """

    def __init__(self, tokenizer=None, max_length=24576, split='train',
                 max_samples=None, max_context_length=None, with_chunks=False,
                 lora_pool_path=None):
        """
        Args:
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length (total)
            split: 'train' or 'validation'
            max_samples: Limit number of samples (for quick testing)
            max_context_length: Max tokens for context (to control long-context vs short)
            with_chunks: if True, pre-compute per-example chunk boundary indices
                         (last-token of each paragraph + question + answer) and
                         attach 'chunk_ends' / 'chunk_names' to every returned item.
                         Requires a fast tokenizer (offset_mapping support).
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_context_length = max_context_length
        self.with_chunks = with_chunks

        # Load dataset (canonical pool if specified, else HF)
        self.examples = []
        skipped_no_answer = 0
        skipped_too_long = 0

        if lora_pool_path is not None and split == 'train':
            print(f"Loading HotpotQA from canonical pool {lora_pool_path}")
            import json as _json
            with open(lora_pool_path) as _f:
                pool = _json.load(_f)
            pool_examples = pool['examples'] if isinstance(pool, dict) and 'examples' in pool else pool
            print(f"  Loaded {len(pool_examples)} canonical examples")
            iter_source = pool_examples
            n_canonical = True
        else:
            print(f"Loading HotpotQA {split} split...")
            dataset = load_dataset('hotpot_qa', 'distractor', split=split)
            iter_source = dataset
            n_canonical = False

        for item in iter_source:
            # Stop once we have enough valid samples
            if max_samples is not None and len(self.examples) >= max_samples:
                break

            question = item['question']
            answer = item['answer']

            # Skip examples without answer
            if not answer or answer.strip() == '':
                skipped_no_answer += 1
                continue

            # Get context paragraphs
            context = list(zip(item['context']['title'], item['context']['sentences']))

            # Create formatted prompt
            full_text = format_hotpotqa_prompt(question, context, answer)

            # Check actual token length — skip if exceeds max_length (no truncation)
            token_len = len(self.tokenizer.encode(full_text, add_special_tokens=True))
            if token_len > max_length:
                skipped_too_long += 1
                continue

            self.examples.append({
                'question': question,
                'answer': answer,
                'context': context,
                'full_text': full_text,
                'question_type': item.get('type', 'unknown'),
            })

        print(f"  Loaded {len(self.examples)} examples")
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
            # Question+answer alone exceeds max_length, truncate everything
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

            # Mask everything except last 20 tokens (best effort to keep answer)
            labels = input_ids.clone()
            labels[:-20] = -100
            labels[labels == self.tokenizer.pad_token_id] = -100

        else:
            # Build context from paragraphs
            context_parts = []
            for title, sentences in example['context']:
                paragraph = ' '.join(sentences)
                context_parts.append(f"{title}: {paragraph}")
            context_str = '\n\n'.join(context_parts)

            # Truncate context to fit
            context_encoded = self.tokenizer(
                f"Context:\n{context_str}",
                add_special_tokens=False,
                truncation=True,
                max_length=available_for_context,
                padding=False,
            )

            # Combine context + question + answer + EOS
            eos = [self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id is not None else []
            combined_ids = context_encoded['input_ids'] + question_encoded['input_ids'] + eos

            # Add BOS token if it exists
            if self.tokenizer.bos_token_id is not None:
                combined_ids = [self.tokenizer.bos_token_id] + combined_ids

            input_ids = torch.tensor(combined_ids, dtype=torch.long)
            attention_mask = torch.ones_like(input_ids)

            # Double-check length
            if len(input_ids) > self.max_length:
                input_ids = input_ids[:self.max_length]
                attention_mask = attention_mask[:self.max_length]

            # Create labels: mask context+question, keep only answer
            labels = input_ids.clone()

            # Find where answer starts (after "Answer: ")
            answer_tokens = self.tokenizer.encode(example['answer'], add_special_tokens=False)
            answer_plus_eos = len(answer_tokens) + len(eos)
            answer_start = len(input_ids) - answer_plus_eos

            # Mask everything before answer
            labels[:answer_start] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }

    def _getitem_with_chunks(self, idx):
        """Alternative getitem: tokenize full_text with offset_mapping, detect chunks.

        Returns the same 3 tensors plus:
            chunk_ends  : List[int]  (per-chunk last-token indices)
            chunk_names : List[str]  ('P1', 'P2', ..., 'Q', 'A')
        """
        import re
        # Late import to avoid circular (data/ doesn't need this normally)
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

        # Detect chunks. Truncate the A chunk to just "Answer:" (no answer body)
        # to match eval-time chunk content — eliminates the train/eval mismatch on
        # A's last-token-pool. With this, generated answer tokens at eval inherit
        # the same A-chunk gate that training uses for its answer tokens.
        answer_marker = re.search(r"\nAnswer:", text)
        if answer_marker is None:
            raise RuntimeError(f"No 'Answer:' in text for idx={idx}")
        answer_end_char = answer_marker.end()  # position right after the colon
        chunks = detect_hotpotqa_chunks(text, offsets, answer_end_char=answer_end_char)
        chunk_ends = [c['end_tok'] for c in chunks]
        chunk_names = [c['name'] for c in chunks]

        # Build labels: mask everything except answer region
        answer_match = re.search(r"\nAnswer:\s+", text)
        if answer_match is None:
            raise RuntimeError(f"No 'Answer:' in text for idx={idx}")
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
    # Test data loader
    from transformers import AutoTokenizer

    print("Testing HotpotQA data loader...")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    tokenizer.pad_token = tokenizer.eos_token

    # Create dataset (use small max_samples for testing)
    dataset = HotpotQADataset(
        tokenizer=tokenizer,
        max_length=2048,
        split='train',
        max_samples=5,
    )

    # Test first example
    example = dataset[0]
    print(f"\nFirst example:")
    print(f"  Input IDs shape: {example['input_ids'].shape}")
    print(f"  Attention mask shape: {example['attention_mask'].shape}")
    print(f"  Labels shape: {example['labels'].shape}")

    # Show details
    print(f"\n  Question: {dataset.examples[0]['question']}")
    print(f"  Answer: {dataset.examples[0]['answer']}")
    print(f"  Context paragraphs: {len(dataset.examples[0]['context'])}")

    # Check label masking
    n_masked = (example['labels'] == -100).sum().item()
    n_total = len(example['labels'])
    n_valid = n_total - n_masked
    print(f"\n  Valid label tokens: {n_valid}/{n_total} ({100*n_valid/n_total:.1f}%)")
    print(f"  (Should have valid tokens for training!)")

    print("\n✅ Tests passed!")
