#!/usr/bin/env python3
#
# Debug utilities for CLaRa training
# Provides color-coded token-by-token output based on model confidence scores
#

import re
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812


def _ansi_fg_rgb(r: int, g: int, b: int) -> str:
    """Generate ANSI escape code for RGB foreground color."""
    r = max(0, min(255, int(r)))
    g = max(0, min(255, int(g)))
    b = max(0, min(255, int(b)))
    return f"\x1b[38;2;{r};{g};{b}m"


def _ansi_reset() -> str:
    """Reset ANSI formatting."""
    return "\x1b[0m"


def _score_to_rgb(score: float) -> tuple[int, int, int]:
    """Convert confidence score (0-1) to RGB color.

    score=0 -> red (low confidence)
    score=1 -> green (high confidence)
    """
    score = float(score)
    if score < 0.0:
        score = 0.0
    elif score > 1.0:
        score = 1.0

    # Color = score*green + (1-score)*red
    r = round(255 * (1.0 - score))
    g = round(255 * score)
    b = 0
    return r, g, b


def _decode_to_str(tokenizer: Any, ids: Any) -> str:
    """Decode token IDs to string."""
    decoded = tokenizer.decode(ids, skip_special_tokens=False)
    if isinstance(decoded, list):
        return " ".join(decoded)
    return str(decoded)


def _preview_text(s: str, *, max_chars: int = 600) -> str:
    """Preview text with truncation."""
    s = str(s)
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + f"…(+{len(s) - max_chars} chars)"


def _compact_mem_tokens_in_prompt(prompt_text: str) -> str:
    """Compact <MEM0><MEM1>... runs to <MEM...>*N for debug printing."""

    # Match sequences of memory tokens
    def repl(match: re.Match[str]) -> str:
        count = len(re.findall(r"<MEM\d*>", match.group(0)))
        if count > 0:
            return f"<MEM...>*{count}"
        return match.group(0)

    # Replace sequences of MEM tokens
    return re.sub(r"(?:<MEM\d*>)+", repl, prompt_text)


def _append_table(lines: list[str], title: str, rows: list[tuple[str, str]]) -> None:
    """Append a simple aligned key/value table into the log buffer."""
    if not rows:
        return
    lines.append(title)
    pad = max(len(k) for k, _ in rows)
    for k, v in rows:
        lines.append(f"  {k.ljust(pad)} : {v}")


@torch.no_grad()
def debug_clara_training(
    model: Any,
    batch: dict[str, Any],
    outputs: dict[str, Any],
    sample_idx: int = 0,
    max_print_tokens: int = 100,
    verbose: bool = True,
) -> None:
    """Debug CLaRa training with color-coded token-by-token output.

    Shows answer tokens colored by model's probability (red=low, green=high).

    Args:
        model: CLaRa model
        batch: Training batch containing input_ids, labels, etc.
        outputs: Model outputs containing logits
        sample_idx: Which sample in the batch to debug
        max_print_tokens: Maximum tokens to display
        verbose: Whether to show detailed information
    """
    log: list[str] = []
    tokenizer = model.decoder_tokenizer

    # Get the decoder inputs and labels
    dec_input_ids = batch.get("dec_input_ids")
    labels = batch.get("labels")

    if dec_input_ids is None or labels is None:
        print("[Debug] Cannot debug: missing dec_input_ids or labels in batch")
        return

    # Get logits from outputs
    logits = outputs.get("logits")
    if logits is None:
        print("[Debug] Cannot debug: missing logits in outputs")
        return

    # Work with one sample
    if sample_idx >= dec_input_ids.size(0):
        sample_idx = 0

    input_ids = dec_input_ids[sample_idx].detach().cpu()
    sample_labels = labels[sample_idx].detach().cpu()
    sample_logits = logits[sample_idx].detach().cpu().float()

    # Get gold token IDs (where labels != -100)
    gold_ids_list = input_ids.tolist()

    # Shift logits for next-token prediction
    shifted_logits = sample_logits[:-1, :]  # Remove last position
    gold_next = gold_ids_list[1:]  # Next tokens to predict

    # Compute predictions
    pred_next = shifted_logits.argmax(dim=-1).tolist()

    # Compute probabilities for gold tokens
    gold_scores: list[float] = []
    if gold_next:
        probs = torch.softmax(shifted_logits, dim=-1)
        gold_idx = torch.tensor(gold_next, dtype=torch.long)
        gold_scores_t = probs.gather(-1, gold_idx.unsqueeze(-1)).squeeze(-1)
        gold_scores = gold_scores_t.tolist()

    # Find answer region (where labels != -100)
    label_mask = sample_labels != -100

    if label_mask.any():
        valid_indices = label_mask.nonzero(as_tuple=True)[0].tolist()
        # Match predictions with gold (accounting for shift by 1)
        matches = []
        for i in valid_indices:
            if i > 0 and (i - 1) < len(pred_next) and (i - 1) < len(gold_next):
                matches.append(int(pred_next[i - 1] == gold_next[i - 1]))
        acc = (sum(matches) / len(matches)) if matches else 0.0
    else:
        matches = []
        acc = 0.0

    # Compute CE loss on answer tokens
    if label_mask.any():
        valid_shifted_mask = label_mask[1:]  # Shift mask for logits alignment
        if valid_shifted_mask.any():
            answer_logits = shifted_logits[valid_shifted_mask]
            answer_labels = sample_labels[1:][valid_shifted_mask]
            ce_loss = F.cross_entropy(answer_logits, answer_labels).item()
        else:
            ce_loss = 0.0
    else:
        ce_loss = 0.0

    # Build debug output
    log.append("\n" + "=" * 80)
    log.append(f"[Debug] CLaRa Training Debug - Sample {sample_idx}")
    log.append("=" * 80)

    # Show question and answer if available
    questions = batch.get("questions")
    if isinstance(questions, list) and sample_idx < len(questions):
        q = str(questions[sample_idx]).strip()
        if q:
            log.append(f"[Debug] Question: {_preview_text(q, max_chars=400)}")

    answers = batch.get("answers")
    gold_answer = ""
    if isinstance(answers, list) and sample_idx < len(answers):
        gold_answer = str(answers[sample_idx]).strip()
        if gold_answer:
            log.append(f"[Debug] Expected Answer: {_preview_text(gold_answer, max_chars=300)}")

    # Show metrics
    _append_table(
        log,
        "[Debug] Answer metrics",
        [
            ("token_acc", f"{acc * 100:.2f}% ({sum(matches)}/{len(matches)})"),
            ("ce_loss", f"{ce_loss:.6f}"),
        ],
    )

    # Color-coded output: show answer tokens colored by probability
    if label_mask.any():
        answer_start_idx = label_mask.nonzero(as_tuple=True)[0][0].item()

        # Prompt (uncolored, compacted)
        prompt_ids = gold_ids_list[:answer_start_idx]
        prompt_text = _decode_to_str(tokenizer, prompt_ids)
        prompt_text = _compact_mem_tokens_in_prompt(prompt_text)

        # Answer tokens (colored by model probability)
        answer_ids = gold_ids_list[answer_start_idx:]
        # Score index: account for shift (logits predict next token)
        answer_scores_start = answer_start_idx - 1  # Because logits are shifted

        colored_parts: list[str] = []
        n = min(len(answer_ids), max_print_tokens)

        for i in range(n):
            tok_id = answer_ids[i]
            tok_txt = _decode_to_str(tokenizer, [tok_id])
            score_idx = answer_scores_start + i

            if 0 <= score_idx < len(gold_scores):
                score = gold_scores[score_idx]
                r, g, b = _score_to_rgb(score)
                colored_parts.append(f"{_ansi_fg_rgb(r, g, b)}{tok_txt}{_ansi_reset()}")
            else:
                colored_parts.append(tok_txt)

        suffix = ""
        if n < len(answer_ids):
            suffix = f"{_ansi_reset()}…(+{len(answer_ids) - n} tokens)"

        log.append(
            "\n[Debug] Answer colored by P(gold token) - red=low confidence, green=high confidence:\n"
            f"{_preview_text(prompt_text, max_chars=300)}{''.join(colored_parts)}{suffix}\n"
        )

    log.append("=" * 80 + "\n")
    print("\n".join(log))


@torch.no_grad()
def debug_generation(
    model: Any,
    questions: list[str],
    documents: list[list[str]],
    gold_answers: list[str],
    max_new_tokens: int = 128,
) -> None:
    """Debug generation with color-coded output.

    Args:
        model: CLaRa model
        questions: List of questions
        documents: List of document lists per question
        gold_answers: Expected answers
        max_new_tokens: Max tokens to generate
    """
    log: list[str] = []
    log.append("\n" + "=" * 80)
    log.append("[Debug] Generation Debug")
    log.append("=" * 80)

    # Generate
    predictions = model.generate_from_text(
        questions=questions, documents=documents, max_new_tokens=max_new_tokens
    )

    for i, (q, pred, gold) in enumerate(zip(questions, predictions, gold_answers, strict=False)):
        log.append(f"\n[Sample {i}]")
        log.append(f"  Question: {_preview_text(q, max_chars=200)}")
        log.append(f"  Expected: {_preview_text(gold, max_chars=200)}")
        log.append(f"  Generated: {_preview_text(pred, max_chars=200)}")

        # Simple match check
        pred_lower = pred.lower().strip()
        gold_lower = gold.lower().strip()
        is_match = gold_lower in pred_lower or pred_lower == gold_lower
        log.append(f"  Match: {is_match}")

    log.append("=" * 80 + "\n")
    print("\n".join(log))


def print_decode(tokenizer: Any, x: torch.Tensor) -> None:
    """Debug-print a token sequence using the provided tokenizer."""
    if x.ndim == 2:
        x = x[0]

    ids = x.detach().to("cpu").tolist()
    decoded = tokenizer.decode(ids, skip_special_tokens=False)
    tokens = tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=False)
    token_view = " ".join(f"{tok}" for tok in tokens[:50])  # Limit tokens shown

    print(f"\x1b[94m[Debug Decode] {decoded[:500]}\x1b[0m")
    print(f"\x1b[90m[Debug Tokens] {token_view}...\x1b[0m")
