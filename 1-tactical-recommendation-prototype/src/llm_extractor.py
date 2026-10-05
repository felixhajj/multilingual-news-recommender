import json
import os
from pathlib import Path

from src.extraction_data import (EXTRACTION_SCHEMA_V2, FILTER_CATEGORIES, SYSTEM_PROMPT,
                                 categories_for_schema)
from src.entity_enrichment import enrich_article
from src.checkpoint_loading import checkpoint_loading


def parse_first_json_object(text):
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("The model output did not contain a valid JSON object")


def validate_extraction_response(response, schema_version=EXTRACTION_SCHEMA_V2):
    categories = categories_for_schema(schema_version)
    expected_fields = set(categories) | {"relationships"}
    if not isinstance(response, dict) or set(response) != expected_fields:
        raise ValueError(f"Extraction response must contain exactly {sorted(expected_fields)}")

    for category in categories:
        values = response[category]
        if not isinstance(values, list) or any(
            not isinstance(value, str) or not value.strip() for value in values
        ):
            raise ValueError(f"{category} must be a list of non-empty strings")

    relationships = response["relationships"]
    if not isinstance(relationships, list):
        raise ValueError("relationships must be a list")
    for relationship in relationships:
        if not isinstance(relationship, dict) or set(relationship) != {
            "subject",
            "relation",
            "object",
        }:
            raise ValueError("Every relationship needs subject, relation, and object")
        if any(not isinstance(value, str) or not value.strip() for value in relationship.values()):
            raise ValueError("Relationship values must be non-empty strings")
    return response


def article_extraction_text(article):
    parts = []
    for field in ("title", "summary", "body", "text"):
        value = article.get(field)
        if isinstance(value, str) and value.strip() and value.strip() not in parts:
            parts.append(value.strip())
    if not parts:
        raise ValueError("Article needs text in title, summary, body, or text")
    return "\n\n".join(parts)


def merge_extraction_with_catalogue(article, response, catalogue):
    response = validate_extraction_response(response)
    seeded_article = {
        **article,
        "tags": {category: list(response[category]) for category in FILTER_CATEGORIES},
    }
    enriched = enrich_article(seeded_article, catalogue)
    labels = {
        **{category: enriched["tags"][category] for category in FILTER_CATEGORIES},
        "relationships": response["relationships"],
    }
    return labels, enriched["entities"], enriched["unknown_candidates"]


class ExtractionOutputError(ValueError):
    def __init__(self, message, raw_output):
        super().__init__(message)
        self.raw_output = raw_output


class QwenExtractionAdapter:
    def __init__(self, adapter_path, base_model_name=None, base_revision=None, quantized=None, device=None):
        import torch
        from peft import PeftConfig, PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        if os.getenv("NEWS_CPU_THREADS"):
            torch.set_num_threads(int(os.environ["NEWS_CPU_THREADS"]))

        adapter_path = Path(adapter_path)
        peft_config = PeftConfig.from_pretrained(adapter_path)
        self.base_model_name = base_model_name or peft_config.base_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path, token=False)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        quantized = torch.cuda.is_available() if quantized is None else quantized
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        ) if quantized else None
        device = device or ("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        with checkpoint_loading():
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                revision=base_revision,
                token=False,
                quantization_config=quantization_config,
                device_map={"": device},
                dtype=torch.float32 if device == "cpu" else torch.float16,
            )
        self.model = PeftModel.from_pretrained(base_model, adapter_path)
        self.model.eval()

    def extract(self, text, max_new_tokens=320, system_prompt=SYSTEM_PROMPT,
                schema_version=EXTRACTION_SCHEMA_V2, deadline=None):
        import torch
        import time

        generation_options = {}
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Release processing deadline reached before generation")
            generation_options["max_time"] = remaining

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **generation_options,
            )
        generated_ids = output_ids[0, inputs["input_ids"].shape[1] :]
        self.last_generation_stats = {
            "input_tokens": int(inputs["input_ids"].shape[1]),
            "output_tokens": int(generated_ids.numel()),
            "max_new_tokens": max_new_tokens,
            "ended_with_eos": bool(generated_ids.numel() and int(generated_ids[-1]) == self.tokenizer.eos_token_id),
            "hit_generation_limit": int(generated_ids.numel()) == max_new_tokens,
        }
        raw_output = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        if deadline is not None and time.monotonic() >= deadline:
            error = TimeoutError("Release processing deadline reached during generation")
            error.raw_output = raw_output
            raise error
        try:
            response = validate_extraction_response(
                parse_first_json_object(raw_output), schema_version
            )
        except ValueError as exc:
            raise ExtractionOutputError(str(exc), raw_output) from exc
        return response, raw_output
