"""Dynamic-padding collate shared by the chunked datasets."""
import torch

def collate_chunked(batch, pad_token_id):
    """Collate with dynamic padding. Passes chunk_ends through as a list-of-lists."""
    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids, attn, labels = [], [], []
    for item in batch:
        pad = max_len - len(item["input_ids"])
        input_ids.append(torch.cat([
            item["input_ids"],
            torch.full((pad,), pad_token_id, dtype=item["input_ids"].dtype),
        ]))
        attn.append(torch.cat([
            item["attention_mask"],
            torch.zeros(pad, dtype=item["attention_mask"].dtype),
        ]))
        labels.append(torch.cat([
            item["labels"],
            torch.full((pad,), -100, dtype=item["labels"].dtype),
        ]))
    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attn),
        "labels": torch.stack(labels),
        "chunk_ends": [item["chunk_ends"] for item in batch],
        "chunk_names": [item["chunk_names"] for item in batch],
        "is_supporting": torch.tensor(
            [item["is_supporting"] for item in batch], dtype=torch.float32,
        ),
        "answer_text": [item["answer_text"] for item in batch],
    }
