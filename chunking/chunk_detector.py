"""
Chunk boundary detector for HotpotQA (and other structured prompts).

HotpotQA format after format_hotpotqa_prompt:
    Context:
    <title1>: <sentences>...

    <title2>: <sentences>...
    ...
    <title10>: <sentences>...
    Question: <q>

    Answer: <a><EOS>

Chunks we want:
    - One chunk per context paragraph (the titled sections)
    - One chunk for the question
    - One chunk for the answer

Approach:
    1. Decode input_ids to text with char-level offsets via the tokenizer's
       offset_mapping (returned by fast tokenizers with return_offsets_mapping=True).
    2. Find marker positions in the text via string search:
         "Context:\n", "\n\n", "Question:", "Answer:"
    3. Map those char positions back to token indices via offset_mapping.
    4. Build (chunk_name, start_token_idx, end_token_idx) list.
"""

import re
from typing import List, Tuple, Dict


def detect_hotpotqa_chunks(
    text: str,
    offset_mapping: List[Tuple[int, int]],
    answer_end_char: int = None,
) -> List[Dict]:
    """Find chunk boundaries in a HotpotQA-formatted prompt.

    Args:
        text:           decoded input string.
        offset_mapping: list of (char_start, char_end) per token
                        (from tokenizer(..., return_offsets_mapping=True)).
        answer_end_char: if given, truncate the "A" chunk to end at this char
                        index instead of running to end-of-text. Used at TRAINING
                        time to make the A chunk cover just "Answer:" (no answer
                        body), so its chunk_rep matches what eval sees (where the
                        prompt naturally ends right after "Answer:"). This removes
                        a train/eval mismatch on A's last-token-pool.
    Returns:
        List of dicts, one per chunk: {name, start_tok, end_tok, start_char, end_char, text}
    """
    # Step 1: find the key markers in the text
    # HotpotQA prompt always starts with "\nContext:\n"
    context_match = re.search(r"Context:\s*\n", text)
    question_match = re.search(r"\nQuestion:\s*", text)
    answer_match = re.search(r"\nAnswer:\s*", text)

    assert context_match is not None, "No 'Context:' marker found"
    assert question_match is not None, "No 'Question:' marker found"
    assert answer_match is not None, "No 'Answer:' marker found"

    # Context body runs from end of "Context:\n" to start of "\nQuestion:"
    ctx_body_start = context_match.end()
    ctx_body_end = question_match.start()
    ctx_text = text[ctx_body_start:ctx_body_end]

    # Step 2: split context into paragraphs (separated by "\n\n")
    # Each paragraph starts with "<title>: <sentences>"
    paragraph_spans = []
    cursor = ctx_body_start
    # Use finditer to find each \n\n boundary
    for m in re.finditer(r"\n\n", ctx_text):
        abs_boundary = ctx_body_start + m.start()   # start of the "\n\n"
        paragraph_spans.append((cursor, abs_boundary))
        cursor = ctx_body_start + m.end()
    # Final paragraph: from cursor to ctx_body_end
    if cursor < ctx_body_end:
        paragraph_spans.append((cursor, ctx_body_end))

    # Question chunk: from "\nQuestion:" position to start of "\nAnswer:"
    q_start = question_match.start() + 1   # skip the leading \n
    q_end = answer_match.start()
    # Answer chunk: from "\nAnswer:" to end-of-text (eval) or to answer_end_char
    # (training, when caller passes the position right after the colon to mask the
    # answer body out of the chunk).
    a_start = answer_match.start() + 1
    a_end = answer_end_char if answer_end_char is not None else len(text)

    # Step 3: convert char spans → token indices via offset_mapping
    def char_to_tok_start(char_idx):
        """First token that starts at or after char_idx."""
        for i, (s, e) in enumerate(offset_mapping):
            if s >= char_idx:
                return i
            # token straddles the boundary → include it
            if s < char_idx <= e:
                return i
        return len(offset_mapping) - 1

    def char_to_tok_end(char_idx):
        """Last token that ends at or before char_idx."""
        last = 0
        for i, (s, e) in enumerate(offset_mapping):
            if e <= char_idx:
                last = i
            else:
                break
        return last

    chunks = []
    for i, (cs, ce) in enumerate(paragraph_spans):
        chunks.append({
            "name": f"P{i+1}",
            "start_tok": char_to_tok_start(cs),
            "end_tok": char_to_tok_end(ce),
            "start_char": cs,
            "end_char": ce,
            "text": text[cs:ce].strip(),
        })
    chunks.append({
        "name": "Q",
        "start_tok": char_to_tok_start(q_start),
        "end_tok": char_to_tok_end(q_end),
        "start_char": q_start,
        "end_char": q_end,
        "text": text[q_start:q_end].strip(),
    })
    chunks.append({
        "name": "A",
        "start_tok": char_to_tok_start(a_start),
        "end_tok": char_to_tok_end(a_end),
        "start_char": a_start,
        "end_char": a_end,
        "text": text[a_start:a_end].strip(),
    })

    return chunks


def last_token_indices(chunks: List[Dict]) -> List[int]:
    """Return list of last-token indices for pooling."""
    return [c["end_tok"] for c in chunks]
