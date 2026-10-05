"""Checkpointed, bounded v3 QLoRA experiment; never promotes its own weights."""
import gc
import json
import os
import random
import time
from pathlib import Path

from src.extraction_data import EXTRACTION_SCHEMA_V3
from src.checkpoint_loading import checkpoint_loading
from src.llm_extractor import validate_extraction_response
from src.news_pipeline import EXTRACTION_PROMPT_V3
from src.portfolio_config import (ARTIFACTS, BASE_MODEL, BASE_REVISION, configure_cache,
                                  digest, file_digest, read_json, write_json, write_jsonl)


def training_identity(examples, steps, initial_adapter=None):
    if not examples or len(examples) > 500 or len({r["article_id"] for r in examples}) != len(examples):
        raise ValueError("Expected 1-500 unique examples")
    for row in examples:
        if row.get("split") != "train" or row.get("schema_version") != EXTRACTION_SCHEMA_V3:
            raise ValueError("Only explicitly versioned v3 training rows are allowed")
        validate_extraction_response(row["labels"], EXTRACTION_SCHEMA_V3)
        if any(v not in row["text"] for key, values in row["labels"].items() if key != "relationships" for v in values):
            raise ValueError("Nonliteral training entity")
        if any(rel[k] not in row["text"] for rel in row["labels"]["relationships"] for k in ("subject", "object")):
            raise ValueError("Nonliteral training relationship")
    return {"base_model": BASE_MODEL, "base_revision": BASE_REVISION,
            "implementation_sha256": file_digest(__file__),
            "checkpoint_backend": os.getenv("NEWS_SAFETENSORS_BACKEND", "mmap"),
            "tokenizer_revision": BASE_REVISION, "schema_version": EXTRACTION_SCHEMA_V3,
            "prompt_sha256": digest(EXTRACTION_PROMPT_V3), "dataset_hash": digest(examples),
            "dataset_examples": len(examples), "max_steps": steps, "seed": 42,
            "max_sequence_length": 1536, "gradient_accumulation": 4, "learning_rate": 0.0001,
            "checkpoint_every_updates": 8,
            "lora": {"r": 16, "alpha": 32, "dropout": 0.05, "targets": ["q_proj", "k_proj", "v_proj", "o_proj"]},
            "initial_adapter_sha256": file_digest(Path(initial_adapter) / "adapter_model.safetensors") if initial_adapter else None,
            "initial_adapter_config_sha256": file_digest(Path(initial_adapter) / "adapter_config.json") if initial_adapter else None,
            "label_source": "machine_assisted_weak_v3_not_human_gold",
            "exhaustiveness": "unverified; empty arrays are weak teacher judgments, not published gold negatives",
            "loss_definition": "arithmetic mean of four supervised-token mean microbatch losses per optimizer update"}


def assert_resume_identity(existing, identity):
    if existing and existing.get("identity") != identity:
        raise ValueError("Dataset, base, prompt or training settings changed: use a new run ID")


def sample_indices(size, step, accumulation=4, seed=42):
    indices = []
    for position in range(step * accumulation, (step + 1) * accumulation):
        epoch, cursor = divmod(position, size)
        order = list(range(size))
        random.Random(seed + epoch).shuffle(order)
        indices.append(order[cursor])
    return indices


def save_checkpoint(directory, model, state):
    import torch
    checkpoint = directory / "checkpoints" / f"step-{state['step']:04d}"
    model.save_pretrained(checkpoint / "adapter")
    torch.save(state, checkpoint / "state.pt")
    pointer = {"directory": str(checkpoint.relative_to(directory)),
               "state_sha256": file_digest(checkpoint / "state.pt"), "step": state["step"],
               "adapter_sha256": file_digest(checkpoint / "adapter/adapter_model.safetensors"),
               "adapter_config_sha256": file_digest(checkpoint / "adapter/adapter_config.json")}
    write_json(directory / "checkpoint.json", pointer)
    return pointer


def check_checkpoint(directory, pointer):
    checkpoint = directory / pointer["directory"]
    if checkpoint.resolve().parent.parent != directory.resolve():
        raise ValueError("Checkpoint path outside run directory")
    for field, path in (("state_sha256", "state.pt"),
                        ("adapter_sha256", "adapter/adapter_model.safetensors"),
                        ("adapter_config_sha256", "adapter/adapter_config.json")):
        if file_digest(checkpoint / path) != pointer[field]:
            raise ValueError("Durable checkpoint checksum mismatch")
    return checkpoint


def train_v3(run_id, examples, steps=48, initial_adapter=None, seconds=1800, progress=None,
             dataset_artifact="phase3/weak_labels_v3/examples.jsonl"):
    if Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("Expected plain run ID")
    configure_cache()
    identity = training_identity(examples, steps, initial_adapter)
    directory = ARTIFACTS / "runs" / run_id
    manifest_path = directory / "manifest.json"
    manifest = read_json(manifest_path, {})
    assert_resume_identity(manifest, identity)
    if manifest.get("status") == "completed":
        if file_digest(directory / "adapter/adapter_model.safetensors") != manifest["adapter_sha256"]:
            raise ValueError("Completed adapter changed")
        return directory / "adapter"
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    if not torch.cuda.is_available():
        raise RuntimeError("New QLoRA experiments require CUDA")
    if os.getenv("NEWS_CPU_THREADS"):
        torch.set_num_threads(int(os.environ["NEWS_CPU_THREADS"]))
    started = time.monotonic()
    manifest = {**manifest, "run_id": run_id, "identity": identity, "status": "running",
                "dataset_artifact": dataset_artifact,
                "promotion_status": "not_evaluated", "runtime_sessions": manifest.get("runtime_sessions", [])}
    write_json(manifest_path, manifest)
    model = optimizer = None
    try:
        torch.manual_seed(42)
        torch.cuda.reset_peak_memory_stats()
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_REVISION, token=False)
        tokenizer.pad_token = tokenizer.eos_token
        with checkpoint_loading():
            base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, revision=BASE_REVISION, token=False,
                device_map={"": 0}, dtype=torch.float16, quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16))
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True,
                                              gradient_checkpointing_kwargs={"use_reentrant": False})
        pointer = read_json(directory / "checkpoint.json")
        checkpoint = check_checkpoint(directory, pointer) if pointer else None
        source = checkpoint / "adapter" if checkpoint else initial_adapter
        model = PeftModel.from_pretrained(base, source, is_trainable=True) if source else get_peft_model(base,
            LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                       target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], bias="none", task_type="CAUSAL_LM"))
        model.config.use_cache = False
        parameters = {name: p for name, p in model.named_parameters() if p.requires_grad}
        initial_path = directory / "initial_weights.pt"
        if not initial_path.exists():
            temporary = initial_path.with_suffix(".tmp")
            torch.save({name: p.detach().cpu().clone() for name, p in parameters.items()}, temporary)
            temporary.replace(initial_path)
        initial_hash = file_digest(initial_path)
        if manifest.get("initial_weights_sha256", initial_hash) != initial_hash:
            raise ValueError("Initial weight evidence changed")
        manifest["initial_weights_sha256"] = initial_hash
        write_json(manifest_path, manifest)
        encoded, dropped = [], []
        for row in examples:
            prompt = tokenizer.apply_chat_template([{"role": "system", "content": EXTRACTION_PROMPT_V3},
                {"role": "user", "content": row["text"]}], tokenize=True, add_generation_prompt=True)
            answer = tokenizer.encode(json.dumps(row["labels"], ensure_ascii=False) + tokenizer.eos_token,
                                      add_special_tokens=False)
            if len(prompt) + len(answer) > identity["max_sequence_length"]:
                dropped.append({"article_id": row["article_id"], "tokens": len(prompt) + len(answer)})
            else:
                encoded.append((row["article_id"], prompt + answer, [-100] * len(prompt) + answer))
        write_jsonl(directory / "dropped_length.jsonl", dropped)
        if len(encoded) < 12:
            raise ValueError("Fewer than 12 complete examples fit; no truncated labels trained")
        optimizer = torch.optim.AdamW(list(parameters.values()), lr=identity["learning_rate"])
        state = {"step": 0, "seen": [], "tokens": 0, "log": [], "identity_hash": digest(identity)}
        if checkpoint:
            state = torch.load(checkpoint / "state.pt", map_location="cpu", weights_only=False)
            if state["identity_hash"] != digest(identity):
                raise ValueError("Checkpoint identity mismatch")
            optimizer.load_state_dict(state["optimizer"])
            torch.set_rng_state(state["rng_cpu"])
            torch.cuda.set_rng_state_all(state["rng_cuda"])
        model.train()
        while state["step"] < steps and time.monotonic() - started < seconds:
            optimizer.zero_grad(set_to_none=True)
            losses, batch_ids, tokens = [], [], 0
            for index in sample_indices(len(encoded), state["step"]):
                article_id, ids, labels = encoded[index]
                inputs = torch.tensor([ids], device="cuda")
                targets = torch.tensor([labels], device="cuda")
                loss = model(input_ids=inputs, attention_mask=torch.ones_like(inputs), labels=targets).loss
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite loss")
                (loss / 4).backward()
                losses.append(float(loss.detach()))
                batch_ids.append(article_id)
                tokens += sum(t != -100 for t in labels)
            torch.nn.utils.clip_grad_norm_(list(parameters.values()), 1.0, error_if_nonfinite=True)
            optimizer.step()
            state["step"] += 1
            state["seen"] = sorted(set(state["seen"]) | set(batch_ids))
            state["tokens"] += tokens
            state["log"].append({"step": state["step"], "loss": sum(losses) / len(losses),
                                 "article_ids": batch_ids, "supervised_tokens": tokens})
            # Commit adapter, optimizer, RNG and cursor together before advancing the pointer.
            if state["step"] % 8 == 0 or state["step"] == steps or time.monotonic() - started >= seconds:
                state.update(optimizer=optimizer.state_dict(), rng_cpu=torch.get_rng_state(),
                             rng_cuda=torch.cuda.get_rng_state_all())
                save_checkpoint(directory, model, state)
                write_jsonl(directory / "loss.jsonl", state["log"])
            if progress:
                progress(f"{run_id}: update {state['step']}/{steps}, loss {state['log'][-1]['loss']:.4f}")
        initial = torch.load(initial_path, map_location="cpu", weights_only=True)
        changes = [{"name": name, "max_abs_change": float((p.detach().cpu() - initial[name]).abs().max())}
                   for name, p in parameters.items()]
        changed = sum(row["max_abs_change"] > 0 for row in changes)
        if not state["step"] or not changed:
            raise RuntimeError("No optimizer updates changed adapter weights")
        model.save_pretrained(directory / "adapter")
        tokenizer.save_pretrained(directory / "adapter")
        write_jsonl(directory / "weight_changes.jsonl", changes)
        write_jsonl(directory / "training_article_ids.jsonl", [{"article_id": i} for i in state["seen"]])
        manifest.update(status="completed" if state["step"] == steps else "time_limited",
                        optimizer_steps=state["step"], unique_articles_seen=len(state["seen"]),
                        supervised_tokens=state["tokens"], encoded_sequences=len(encoded), dropped_sequences=len(dropped),
                        trainable_parameters=sum(p.numel() for p in parameters.values()),
                        changed_tensors=changed, total_trainable_tensors=len(parameters),
                        adapter_sha256=file_digest(directory / "adapter/adapter_model.safetensors"),
                        adapter_config_sha256=file_digest(directory / "adapter/adapter_config.json"),
                        tokenizer_files={p.name: file_digest(p) for p in (directory / "adapter").iterdir()
                                         if p.name in {"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt"}},
                        peak_gpu_gib=torch.cuda.max_memory_allocated() / 1024 ** 3)
        return directory / "adapter"
    except Exception as exc:
        manifest.update(status="interrupted_or_failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        manifest["runtime_sessions"].append({"elapsed_seconds": time.monotonic() - started})
        write_json(manifest_path, manifest)
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()
