"""Bounded real QLoRA experiments with complete provenance and no automatic promotion."""
import gc
import json
import random
import re
import time
from collections import Counter, deque
from datetime import datetime, timezone

from src.extraction_data import FILTER_CATEGORIES
from src.llm_extractor import validate_extraction_response
from src.news_pipeline import EXTRACTION_PROMPT
from src.portfolio_config import (ARTIFACTS, BASE_MODEL, configure_cache, digest, file_digest,
                                  read_json, read_jsonl, write_json, write_jsonl)

SYSTEM_HINTS = re.compile(
    r"missile|drone|aircraft|fighter|radar|weapon|tank|frigate|submarine|air defense|"
    r"\u0635\u0627\u0631\u0648\u062e|\u0637\u0627\u0626\u0631|\u0631\u0627\u062f\u0627\u0631|\u0633\u0644\u0627\u062d|\u0645\u0646\u0638\u0648\u0645|\u062f\u0628\u0627\u0628|\u062f\u0641\u0627\u0639",
    re.I,
)
COMPANY_HINTS = re.compile(
    r"company|corporation|manufacturer|contractor|industries|"
    r"\u0634\u0631\u0643\u0629|\u0634\u0631\u0643\u0627\u062a|\u0645\u062c\u0645\u0648\u0639\u0629|\u0635\u0646\u0627\u0639\u0627\u062a",
    re.I,
)


def _round_robin(groups):
    """Yield every item from each language queue, including unequal tails."""
    queues = [deque(group) for group in groups if group]
    while queues:
        remaining = []
        for queue in queues:
            yield queue.popleft()
            if queue:
                remaining.append(queue)
        queues = remaining


def build_labeling_queue(records, accepted, frozen_groups, limit=300):
    """Choose deterministic train-only candidates weighted toward current coverage gaps."""
    accepted_ids = {row["article_id"] for row in accepted}
    candidates = []
    for row in records:
        if (row["split"] != "train" or row["article_id"] in accepted_ids
                or row["group_id"] in frozen_groups or not row["mentions"]
                or not 250 <= len(row["body"]) <= 2400):
            continue
        text = row["title"] + "\n\n" + row["body"]
        reasons = []
        if SYSTEM_HINTS.search(text):
            reasons.append("system_coverage_hint")
        if COMPANY_HINTS.search(text):
            reasons.append("company_coverage_hint")
        if row.get("domain_hits", 0):
            reasons.append("geopolitical_domain_terms")
        score = (2 * ("system_coverage_hint" in reasons)
                 + 2 * ("company_coverage_hint" in reasons)
                 + min(row.get("domain_hits", 0), 20) / 20
                 + min(len(row["mentions"]), 10) / 20)
        candidates.append({"article_id": row["article_id"], "content_hash": row["content_hash"],
                           "group_id": row["group_id"], "language": row["language"],
                           "title": row["title"], "text": text, "source_url": row["source_url"],
                           "source_revision": row["source_revision"], "license_id": row["license_id"],
                           "license_url": row["license_url"], "mentions": row["mentions"],
                           "selection_reasons": reasons or ["published_positive_mentions"],
                           "selection_score": round(score, 4)})
    ordered = []
    for language in ("en", "ar"):
        group = [row for row in candidates if row["language"] == language]
        group.sort(key=lambda row: (-row["selection_score"], digest(row["article_id"])))
        ordered.append(group)
    return list(_round_robin(ordered))[:limit]


def validate_grounded_labels(labels, text):
    """Reject structurally invalid or nonliteral model labels before persistence."""
    validate_extraction_response(labels)
    if any(value not in text for category in FILTER_CATEGORIES for value in labels[category]):
        raise ValueError("nonliteral entity label")
    if any(relation[key] not in text for relation in labels["relationships"]
           for key in ("subject", "object")):
        raise ValueError("nonliteral relationship endpoint")


def quarantine_invalid_additions(additions, attempts, directory):
    """Remove previously accepted invalid rows without retrying their model calls."""
    valid, invalid = [], []
    for row in additions:
        try:
            validate_grounded_labels(row["labels"], row["text"])
            valid.append(row)
        except ValueError as exc:
            invalid.append({**row, "quarantine_reason": str(exc),
                            "quarantined_at": datetime.now(timezone.utc).isoformat()})
    if not invalid:
        return additions, attempts

    invalid_ids = {row["article_id"]: row["quarantine_reason"] for row in invalid}
    quarantine_path = directory / "quarantined.jsonl"
    previous = read_jsonl(quarantine_path) if quarantine_path.exists() else []
    previous_by_id = {row["article_id"]: row for row in previous}
    previous_by_id.update({row["article_id"]: row for row in invalid})
    write_jsonl(quarantine_path, list(previous_by_id.values()))
    write_jsonl(directory / "accepted.jsonl", valid)

    corrected = []
    for attempt in attempts:
        if attempt["article_id"] in invalid_ids and attempt.get("outcome") == "accepted":
            attempt = {**attempt, "original_outcome": "accepted", "outcome": "rejected",
                       "reason": f"Post-acceptance grounding audit: {invalid_ids[attempt['article_id']]}",
                       "corrected_at": datetime.now(timezone.utc).isoformat()}
        corrected.append(attempt)
    write_jsonl(directory / "attempts.jsonl", corrected)
    return valid, corrected


def validate_training_examples(examples, corpus, frozen_groups):
    by_id = {row["article_id"]: row for row in corpus}
    ids = [row["article_id"] for row in examples]
    errors = []
    if len(ids) != len(set(ids)):
        errors.append("duplicate article IDs")
    for row in examples:
        source = by_id.get(row["article_id"])
        if not source:
            errors.append(f"{row['article_id']}: missing corpus source")
            continue
        if row.get("split") != "train" or source["split"] != "train":
            errors.append(f"{row['article_id']}: not train-only")
        if row.get("group_id") in frozen_groups:
            errors.append(f"{row['article_id']}: overlaps a frozen review group")
        if row.get("source_content_hash") != source["content_hash"]:
            errors.append(f"{row['article_id']}: source hash mismatch")
        labels = row.get("labels", {})
        try:
            validate_grounded_labels(labels, row["text"])
        except ValueError as exc:
            errors.append(f"{row['article_id']}: {exc}")
            continue
    if errors:
        raise ValueError("Training data validation failed: " + "; ".join(errors[:20]))
    return {"examples": len(examples), "languages": dict(Counter(row["language"] for row in examples)),
            "nonempty_fields": {category: sum(bool(row["labels"][category]) for row in examples)
                                for category in FILTER_CATEGORIES}}


def prepare_training_examples(records, backend, target=500, seconds=10800, candidate_queue=None):
    baseline_path = ARTIFACTS / "extraction_training.jsonl"
    historical_baseline = read_jsonl(baseline_path) if baseline_path.exists() else []
    phase2_seed = ARTIFACTS / "phase2" / "usable_training_seed.jsonl"
    baseline = read_jsonl(phase2_seed) if phase2_seed.exists() else historical_baseline
    frozen = read_json(ARTIFACTS / "review" / "frozen_extraction_manifest.json", {})
    frozen_groups = set(frozen.get("group_ids", []))
    queue = candidate_queue or build_labeling_queue(records, historical_baseline, frozen_groups)
    policy = {"version": 2, "teacher": backend.identity, "prompt": EXTRACTION_PROMPT,
              "candidate_identity": digest([{k: row[k] for k in ("article_id", "content_hash", "group_id", "language")}
                                             for row in queue]),
              "validation": "strict schema, literal entities, published-positive agreement",
              "retry_policy": "one completed attempt per article/content/teacher/prompt identity"}
    version = digest(policy)
    directory = ARTIFACTS / "labeling" / version
    attempts_path = directory / "attempts.jsonl"
    additions_path = directory / "accepted.jsonl"
    attempts = read_jsonl(attempts_path) if attempts_path.exists() else []
    additions = read_jsonl(additions_path) if additions_path.exists() else []
    additions, attempts = quarantine_invalid_additions(additions, attempts, directory)
    accepted = baseline + additions
    seen = {row["article_id"] for row in accepted}
    completed = {row["attempt_id"] for row in attempts}
    started = time.monotonic()
    for article in queue:
        if len(accepted) >= target or time.monotonic()-started >= seconds:
            break
        attempt_id = digest({"version": version, "article_id": article["article_id"],
                             "content_hash": article["content_hash"]})
        if article["article_id"] in seen or attempt_id in completed:
            continue
        text = article["text"]
        attempt_started = time.monotonic()
        raw = None
        try:
            labels, raw = backend.extract(text)
            validate_grounded_labels(labels, text)
            flattened = [v for c in FILTER_CATEGORIES for v in labels[c]]
            if not flattened:
                raise ValueError("No grounded labels were extracted")
            reference_mentions = {m["text"].casefold() for m in article["mentions"]}
            predicted = {v.casefold() for v in flattened}
            missing_positives = sorted(reference_mentions - predicted)
            if missing_positives:
                raise ValueError(f"Missing {len(missing_positives)} independently published positive mentions")
            addition = {"article_id": article["article_id"], "text": text,
                "labels": labels, "language": article["language"], "split": "train",
                "group_id": article["group_id"], "source_url": article["source_url"],
                "source_content_hash": article["content_hash"], "label_source": "model_assisted",
                "teacher": backend.identity, "review_status": "machine_checked_not_human_reviewed",
                "annotation_completeness": "model_estimate_with_all_published_positives_covered_not_human_gold",
                "published_positive_count": len(reference_mentions),
                "selection_reasons": article["selection_reasons"], "raw_model_outputs": raw}
            additions.append(addition)
            accepted.append(addition)
            seen.add(article["article_id"])
            write_jsonl(additions_path, additions)
            outcome, reason = "accepted", None
            print(f"Accepted {len(accepted)}/{target}: {article['article_id']}", flush=True)
        except (ValueError, RuntimeError) as exc:
            outcome, reason = "rejected", str(exc)[:500]
            raw = raw if raw is not None else getattr(exc, "raw_output", None)
            if "out of memory" in reason.casefold():
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except (ImportError, RuntimeError):
                    pass
        attempt = {"attempt_id": attempt_id, "dataset_version": version,
                   "article_id": article["article_id"], "content_hash": article["content_hash"],
                   "group_id": article["group_id"], "language": article["language"],
                   "teacher": backend.identity, "prompt_sha256": digest(EXTRACTION_PROMPT),
                   "selection_reasons": article["selection_reasons"], "outcome": outcome,
                   "reason": reason, "raw_model_outputs": raw,
                   "elapsed_seconds": round(time.monotonic()-attempt_started, 3),
                   "completed_at": datetime.now(timezone.utc).isoformat()}
        attempts.append(attempt)
        completed.add(attempt_id)
        write_jsonl(attempts_path, attempts)
        manifest = {"version": version, "policy": policy, "historical_baseline_examples": len(historical_baseline),
                    "usable_baseline_examples": len(baseline),
                    "new_examples": len(additions), "total_examples": len(accepted), "target": target,
                    "attempts": len(attempts), "outcomes": dict(Counter(row["outcome"] for row in attempts)),
                    "elapsed_seconds_this_invocation": round(time.monotonic()-started, 3),
                    "status": "running", "label_source": "machine-assisted, not human gold"}
        write_json(directory / "manifest.json", manifest)
    validate_training_examples(accepted, records, frozen_groups)
    manifest = {"version": version, "policy": policy, "historical_baseline_examples": len(historical_baseline),
                "usable_baseline_examples": len(baseline),
                "new_examples": len(additions), "total_examples": len(accepted), "target": target,
                "attempts": len(attempts), "outcomes": dict(Counter(row["outcome"] for row in attempts)),
                "elapsed_seconds_this_invocation": round(time.monotonic()-started, 3),
                "status": "target_reached" if len(accepted) >= target else "bounded_run_finished",
                "dataset_hash": digest(accepted),
                "label_source": "machine-assisted, not independently human reviewed"}
    write_json(directory / "manifest.json", manifest)
    write_json(ARTIFACTS / "training_preparation_v2.json", manifest)
    return accepted


def train_adapter(run_id, examples, stage="sft", initial_adapter=None, max_steps=48, seconds=10800):
    configure_cache()
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA training requires NVIDIA CUDA. Hosted inference does not retrain.")
    if not examples or stage not in {"sft", "domain"}:
        raise ValueError("Provide training examples and a supported stage")
    if any(e.get("split") != "train" for e in examples):
        raise ValueError("Validation and test articles must never enter the trainer")
    directory = ARTIFACTS / "runs" / run_id
    if (directory / "manifest.json").exists():
        raise ValueError("Run IDs are immutable; choose a new ID")
    directory.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    manifest = {"run_id": run_id, "status": "running", "stage": stage, "base_model": BASE_MODEL,
                "started_at": datetime.now(timezone.utc).isoformat(), "max_steps": max_steps,
                "time_budget_seconds": seconds, "initial_adapter": str(initial_adapter) if initial_adapter else None,
                "dataset_hash": digest(examples), "dataset_examples": len(examples), "seed": 42,
                "max_sequence_length": 1024, "gradient_accumulation": 4, "learning_rate": 0.0001,
                "promotion_status": "not_evaluated", "label_source": "raw_articles" if stage == "domain" else "model_assisted_not_human_gold"}
    write_json(directory / "manifest.json", manifest)
    log, steps, seen, trained_tokens = [], 0, set(), 0
    try:
        torch.manual_seed(42)
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=False)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, token=False, device_map={"": 0},
            dtype=torch.float16, quantization_config=BitsAndBytesConfig(load_in_4bit=True,
                bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16))
        manifest["base_revision"] = model.config._commit_hash
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                                gradient_checkpointing_kwargs={"use_reentrant": False})
        model = PeftModel.from_pretrained(model, initial_adapter, is_trainable=True) if initial_adapter else get_peft_model(model,
            LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                       bias="none", task_type="CAUSAL_LM"))
        model.config.use_cache = False
        tracked_name, tracked = next((name, param) for name, param in model.named_parameters() if "lora_B" in name and param.requires_grad)
        initial = tracked.detach().cpu().clone()
        manifest["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        encoded = []
        for example in examples:
            if stage == "domain":
                ids = tokenizer.encode(example["text"], add_special_tokens=False)
                for offset in range(0, len(ids), 1024):
                    segment = ids[offset:offset+1024]
                    if len(segment) >= 64:
                        encoded.append((example["article_id"], segment, segment.copy()))
            else:
                prompt = tokenizer.apply_chat_template([{"role": "system", "content": EXTRACTION_PROMPT},
                    {"role": "user", "content": example["text"]}], tokenize=True, add_generation_prompt=True)
                answer = tokenizer.encode(json.dumps(example["labels"], ensure_ascii=False) + tokenizer.eos_token, add_special_tokens=False)
                if len(prompt) + len(answer) <= 1024:
                    encoded.append((example["article_id"], prompt+answer, [-100]*len(prompt)+answer))
        if not encoded:
            raise ValueError("No complete examples fit the context limit; no truncated labels will be trained")
        manifest["encoded_sequences"] = len(encoded)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.0001)
        model.train()
        optimizer.zero_grad()
        rng = random.Random(42)
        batches = 0
        while steps < max_steps and time.monotonic()-started < seconds:
            rng.shuffle(encoded)
            for article_id, ids, labels in encoded:
                if steps >= max_steps or time.monotonic()-started >= seconds:
                    break
                inputs = torch.tensor([ids], device="cuda")
                targets = torch.tensor([labels], device="cuda")
                loss = model(input_ids=inputs, attention_mask=torch.ones_like(inputs), labels=targets).loss
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite training loss")
                (loss/4).backward()
                batches += 1
                seen.add(article_id)
                trained_tokens += sum(value != -100 for value in labels)
                if batches % 4 == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                    optimizer.step()
                    optimizer.zero_grad()
                    steps += 1
                    entry = {"step": steps, "loss": float(loss.detach()), "elapsed_seconds": round(time.monotonic()-started, 2)}
                    log.append(entry)
                    write_jsonl(directory / "loss.jsonl", log)
                    print(f"{run_id}: {entry}", flush=True)
        delta = float((tracked.detach().cpu()-initial).abs().max())
        if not steps or delta == 0:
            raise RuntimeError("No optimizer update changed the adapter")
        model.save_pretrained(directory / "adapter")
        tokenizer.save_pretrained(directory / "adapter")
        manifest.update(status="completed" if steps == max_steps else "time_limited",
                        optimizer_steps=steps, unique_articles_seen=len(seen), supervised_tokens=trained_tokens,
                        tracked_parameter=tracked_name, maximum_weight_change=delta,
                        adapter_sha256=file_digest(directory / "adapter" / "adapter_model.safetensors"),
                        peak_gpu_gib=torch.cuda.max_memory_allocated()/1024**3)
        write_jsonl(directory / "training_article_ids.jsonl", [{"article_id": identifier} for identifier in sorted(seen)])
        del optimizer, model
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}", optimizer_steps=steps)
        raise
    finally:
        manifest["elapsed_seconds"] = round(time.monotonic()-started, 2)
        write_json(directory / "manifest.json", manifest)
    return directory / "adapter"
