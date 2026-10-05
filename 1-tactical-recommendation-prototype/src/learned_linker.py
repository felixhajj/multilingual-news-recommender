"""Contextual entity retrieval with a supervised, inspectable candidate ranker."""
import math
import inspect
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher

import numpy as np

from src.portfolio_config import (ARTIFACTS, EMBEDDING_MODEL, EMBEDDING_REVISION,
                                  configure_cache, digest, file_digest, read_json,
                                  write_json, write_jsonl)

# Phase 3 trained the scalar implementation. This explicitly approved bridge
# changes only request batching; candidate features, scoring and abstention stay fixed.
LINKER_RUNTIME_COMPATIBILITY = {
    ("d88af23f3f5905b0aed0bab9db4b46b466942d38e12ed594ae0709416ad403fd",
     "014b2903d5c4de77d66e382457d8b2d361701c4fe81b597f69edc481b8e98790"):
        "batched-encoding-same-candidate-scorer-v1",
}


def encode_batch(texts):
    configure_cache()
    from src.embeddings import _get_model
    if not texts:
        return np.empty((0, 768), dtype=np.float32)
    model = _get_model()
    chunks, owners = [], []
    for index, text in enumerate(texts):
        prefix, separator, body = text.partition(": ")
        if not separator:
            prefix, body = "passage", text
        offsets = model.tokenizer(body, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
        if len(offsets) <= 448:
            chunks.append(text)
            owners.append(index)
        else:
            for start in range(0, len(offsets), 416):
                end = min(len(offsets), start+448)
                chunks.append(prefix + ": " + body[offsets[start][0]:offsets[end-1][1]])
                owners.append(index)
    vectors = model.encode(chunks, batch_size=16, normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=False)
    results = np.zeros((len(texts), vectors.shape[1]), dtype=np.float32)
    counts = np.zeros(len(texts), dtype=np.float32)
    for owner, vector in zip(owners, vectors):
        results[owner] += vector
        counts[owner] += 1
    results /= counts[:, None]
    results /= np.maximum(np.linalg.norm(results, axis=1, keepdims=True), 1e-12)
    return results


def normalized(text):
    return " ".join(re.findall(r"\w+", text.casefold()))


def context_at(text, start, end):
    return text[max(0, start-180):min(len(text), end+180)]


def candidate_features(mention, context_vector, mention_vector, entities, vectors):
    context_scores = vectors @ context_vector
    mention_scores = vectors @ mention_vector
    surface = normalized(mention)
    lexical = np.asarray([max(float(surface in e.get("normalized_aliases", [])),
                              SequenceMatcher(None, surface, normalized(e["name"])).ratio())
                          for e in entities], dtype=np.float32)
    return np.column_stack([mention_scores, context_scores, lexical,
                            mention_scores * context_scores]).astype(np.float32)


def linker_artifact_identity(directory):
    """Bind runtime/cache identity to bytes, without upgrading legacy provenance."""
    config = read_json(directory / "model.json")
    runtime_linking = digest(inspect.getsource(LearnedEntityLinker.link)
                             + inspect.getsource(LearnedEntityLinker.score))
    trained_linking = config.get("linking_implementation")
    compatibility = LINKER_RUNTIME_COMPATIBILITY.get((trained_linking, runtime_linking))
    if trained_linking and trained_linking != runtime_linking and compatibility is None:
        raise ValueError("Linker implementation differs from its trained artifact")
    identity = {"model_sha256": file_digest(directory / "model.json"),
                "catalogue_sha256": file_digest(directory / "entities.json"),
                "vectors_sha256": file_digest(directory / "entity_vectors.npy"),
                "encoder": config["encoder"],
                "vector_encoder_revision": config.get("encoder_revision", "unrecorded_in_legacy_artifact"),
                "runtime_encoder_revision": EMBEDDING_REVISION,
                "feature_implementation": digest(inspect.getsource(candidate_features)
                                                  + inspect.getsource(normalized)),
                "encoding_implementation": digest(inspect.getsource(encode_batch)),
                "linking_implementation": trained_linking,
                "runtime_linking_implementation": runtime_linking,
                "linking_compatibility": compatibility}
    expected = config.get("entity_vectors_sha256")
    if expected and expected != identity["vectors_sha256"]:
        raise ValueError("Linker vectors do not match their trained artifact")
    if config.get("encoder_revision", EMBEDDING_REVISION) != EMBEDDING_REVISION:
        raise ValueError("Linker vector encoder revision differs from runtime")
    if config["encoder"] != EMBEDDING_MODEL:
        raise ValueError("Linker encoder differs from runtime")
    for field in ("feature_implementation", "encoding_implementation"):
        if config.get(field) and config[field] != identity[field]:
            raise ValueError("Linker implementation differs from its trained artifact")
    identity["training_identity_complete"] = all(config.get(k) for k in (
        "encoder_revision", "entity_vectors_sha256", "feature_implementation", "encoding_implementation",
        "linking_implementation"))
    return identity


class LearnedEntityLinker:
    def __init__(self, directory=None, encoder=encode_batch):
        deployed = read_json(ARTIFACTS / "active_model.json", {}) if directory is None else {}
        self.directory = directory or ARTIFACTS / deployed.get("linker", "linker")
        if not self.directory.resolve().is_relative_to(ARTIFACTS.resolve()) and directory is None:
            raise ValueError("Active linker path outside artifact directory")
        self.config = read_json(self.directory / "model.json")
        if not self.config:
            raise RuntimeError("Learned linker is unavailable. Run scripts/train_linker.py first.")
        self.entities = read_json(self.directory / "entities.json")
        self.vectors = np.load(self.directory / "entity_vectors.npy", allow_pickle=False)
        if self.config["catalogue_hash"] != digest(self.entities):
            raise ValueError("Linker catalogue does not match its trained artifact")
        if (self.vectors.shape != (len(self.entities), 768)
                or not np.isfinite(self.vectors).all()
                or not np.allclose(np.linalg.norm(self.vectors, axis=1), 1, atol=.001)):
            raise ValueError("Expected normalized finite linker vectors aligned with the catalogue")
        self.encoder = encoder
        self.artifact_identity = linker_artifact_identity(self.directory)
        self.legacy_version = digest(self.config)
        self.version = digest(self.artifact_identity)

    def score(self, features):
        scaled = (features - self.config["mean"]) / self.config["scale"]
        logits = scaled @ np.asarray(self.config["coef"]) + self.config["intercept"]
        return 1 / (1 + np.exp(-np.clip(logits, -50, 50)))

    def link(self, mention, context, candidate_limit=3):
        return self.link_many([(mention, context)], candidate_limit)[0]

    def link_many(self, mentions, candidate_limit=3):
        """Encode a document's mentions together; each still uses the same candidate scorer."""
        if not 1 <= candidate_limit <= 10:
            raise ValueError("candidate_limit must be 1-10")
        if not mentions:
            return []
        if any(not isinstance(name, str) or not name.strip()
               or not isinstance(context, str) for name, context in mentions):
            raise ValueError("Each entity link needs a non-empty name and text context")
        count = len(mentions)
        vectors = self.encoder(["query: " + name for name, _ in mentions]
                               + ["query: " + context for _, context in mentions])
        output = []
        for index, (mention, _) in enumerate(mentions):
            features = candidate_features(mention, vectors[count+index], vectors[index], self.entities, self.vectors)
            candidates = np.argsort(features[:, 0] + features[:, 1] + features[:, 2])[-10:]
            scores = self.score(features[candidates])
            order = np.argsort(scores)[::-1]
            best = int(candidates[order[0]])
            score = float(scores[order[0]])
            linked = score >= self.config["threshold"]
            output.append({"mention": mention, "entity_id": self.entities[best]["qid"] if linked else None,
                           "canonical_name": self.entities[best]["name"] if linked else None,
                           "status": "linked" if linked else "unresolved", "link_score": round(score, 4),
                           "unrounded_link_score": score,
                           "score_note": "Learned candidate score, not a calibrated probability",
                           "linker_version": self.version,
                           "candidates": [{"entity_id": self.entities[int(candidates[j])]["qid"],
                                           "name": self.entities[int(candidates[j])]["name"],
                                           "score": round(float(scores[j]), 4)} for j in order[:candidate_limit]]})
        return output


def catalogue_from_training(records, max_entities=3000):
    """Seed the candidate KB only from training links, never held-out aliases."""
    names, counts, contexts = defaultdict(Counter), Counter(), {}
    for article in records:
        if article["split"] != "train":
            continue
        for mention in article["mentions"]:
            qid = mention["qid"]
            names[qid][mention["text"]] += 3 if article["language"] == "en" else 1
            counts[qid] += 1
            if qid not in contexts or article["language"] == "en":
                contexts[qid] = context_at(article["body"], mention["start"], mention["end"])
    return [{"qid": qid, "name": names[qid].most_common(1)[0][0],
             "aliases": sorted(names[qid]), "normalized_aliases": sorted({normalized(a) for a in names[qid]}),
             "description": contexts[qid],
             "provenance": "training-partition published news hyperlinks and context"}
            for qid, _ in counts.most_common(max_entities)]


def cached_training_vectors(destination, name, texts, batch_size=128):
    """Save identity-bound encoding batches so an interrupted fit can resume."""
    folder = destination / "encoding_cache" / name
    identity = {"inputs": digest(texts), "encoder": EMBEDDING_MODEL,
                "revision": EMBEDDING_REVISION, "implementation": digest(inspect.getsource(encode_batch)),
                "batch_size": batch_size}
    previous = read_json(folder / "manifest.json")
    if previous is not None and previous["identity"] != identity:
        raise ValueError("Cached linker training inputs or encoder changed")
    folder.mkdir(parents=True, exist_ok=True)
    manifest = previous or {"identity": identity, "batches": {}}
    write_json(folder / "manifest.json", manifest)
    arrays = []
    for start in range(0, len(texts), batch_size):
        name = f"{start:06d}.npy"
        path = folder / name
        if name in manifest["batches"]:
            if file_digest(path) != manifest["batches"][name]:
                raise ValueError("Cached linker vectors changed")
            vectors = np.load(path, allow_pickle=False)
        else:
            from src.phase3_resources import check_storage
            check_storage(ARTIFACTS)
            vectors = encode_batch(texts[start:start+batch_size])
            np.save(path, vectors, allow_pickle=False)
            manifest["batches"][name] = file_digest(path)
            write_json(folder / "manifest.json", manifest)
        if (vectors.shape != (len(texts[start:start+batch_size]), 768)
                or not np.isfinite(vectors).all()
                or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=.001)):
            raise ValueError("Invalid cached training embeddings")
        arrays.append(vectors)
    return np.concatenate(arrays) if arrays else np.empty((0, 768), dtype=np.float32)


def train_linker(records, max_train=2000, max_eval=300, directory=None, evaluate_test=False):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    destination = directory or ARTIFACTS / "linker"
    destination.mkdir(parents=True, exist_ok=True)
    training_identity = {"records": digest([r for r in records if evaluate_test or r["split"] != "test"]),
                         "max_train": max_train, "max_eval": max_eval, "evaluate_test": evaluate_test,
                         "encoder_revision": EMBEDDING_REVISION,
                         "implementation": digest(inspect.getsource(train_linker)
                                                  + inspect.getsource(cached_training_vectors)
                                                  + inspect.getsource(catalogue_from_training))}
    previous = read_json(destination / "training_inputs.json")
    if previous is not None and previous != training_identity:
        raise ValueError("Linker training configuration changed; use a new run directory")
    write_json(destination / "training_inputs.json", training_identity)
    completed = read_json(destination / "training_complete.json")
    if completed is not None:
        if completed["identity"] != training_identity:
            raise ValueError("Completed linker training identity changed")
        for name, checksum in completed["files"].items():
            if file_digest(destination / name) != checksum:
                raise ValueError("Completed linker training artifacts changed")
        print("Reusing completed linker training", flush=True)
        return read_json(destination / "metrics.json")
    entities = catalogue_from_training(records)
    if len(entities) < 10:
        raise ValueError("Not enough distinct training entities")
    qid_index = {e["qid"]: i for i, e in enumerate(entities)}
    write_json(destination / "entities.json", entities)
    print(f"Encoding {len(entities)} entity descriptions", flush=True)
    entity_vectors = cached_training_vectors(destination, "entities",
        [f"passage: {e['name']}. {e['description']}" for e in entities])
    np.save(destination / "entity_vectors.npy", entity_vectors, allow_pickle=False)
    rows = defaultdict(list)
    for article in sorted(records, key=lambda r: digest(r["article_id"])):
        if article["split"] == "test" and not evaluate_test:
            continue
        for mention in article["mentions"][:5]:
            limit = max_train if article["split"] == "train" else max_eval
            if len(rows[article["split"]]) < limit:
                rows[article["split"]].append({**mention, "article_id": article["article_id"],
                                              "language": article["language"],
                                              "context": context_at(article["body"], mention["start"], mention["end"])})
    features_by_split = {}
    for split, mentions in rows.items():
        print(f"Encoding {len(mentions)} {split} mention contexts", flush=True)
        vectors = cached_training_vectors(destination, split,
            ["query: " + m["text"] for m in mentions] + ["query: " + m["context"] for m in mentions])
        prepared = []
        for i, mention in enumerate(mentions):
            features = candidate_features(mention["text"], vectors[len(mentions)+i], vectors[i], entities, entity_vectors)
            candidates = list(np.argsort(features[:, 0] + features[:, 1] + features[:, 2])[-10:])
            gold = qid_index.get(mention["qid"])
            # Only training gets an injected positive; evaluation measures retrieval misses.
            if split == "train" and gold is not None and gold not in candidates:
                candidates.append(gold)
            prepared.append((mention, np.asarray(candidates), features[candidates], gold))
        features_by_split[split] = prepared
    training = features_by_split["train"]
    x = np.concatenate([item[2] for item in training])
    y = np.concatenate([(item[1] == item[3]).astype(int) for item in training])
    scaler = StandardScaler().fit(x)
    classifier = LogisticRegression(max_iter=500, random_state=42, class_weight="balanced").fit(scaler.transform(x), y)
    scorer = object.__new__(LearnedEntityLinker)
    scorer.config = {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
                     "coef": classifier.coef_[0].tolist(), "intercept": float(classifier.intercept_[0])}
    def predictions(split):
        output = []
        for mention, candidates, features, gold in features_by_split[split]:
            scores = scorer.score(features)
            best = int(np.argsort(scores)[::-1][0])
            alias = [e["qid"] for e in entities if normalized(mention["text"]) in e["normalized_aliases"]]
            output.append({**mention, "predicted_qid": entities[int(candidates[best])]["qid"],
                           "score": float(scores[best]), "gold_in_catalogue": gold is not None,
                           "gold_retrieved": gold in candidates if gold is not None else False,
                           "alias_prediction": alias[0] if len(alias) == 1 else None})
        return output
    validation = predictions("validation")
    threshold = 1.0
    for candidate in np.linspace(0.5, 0.995, 100):
        accepted = [p for p in validation if p["score"] >= candidate]
        if len(accepted) >= 10 and sum(p["predicted_qid"] == p["qid"] for p in accepted) / len(accepted) >= 0.9:
            threshold = float(candidate)
            break
    config = {"encoder": EMBEDDING_MODEL, "encoder_revision": EMBEDDING_REVISION,
              "entity_vectors_sha256": file_digest(destination / "entity_vectors.npy"),
              "feature_implementation": digest(inspect.getsource(candidate_features) + inspect.getsource(normalized)),
              "encoding_implementation": digest(inspect.getsource(encode_batch)),
              "linking_implementation": digest(inspect.getsource(LearnedEntityLinker.link)
                                                + inspect.getsource(LearnedEntityLinker.score)),
              "feature_version": 2, "catalogue_hash": digest(entities),
              **scorer.config,
              "threshold": threshold, "training_mentions": len(training), "training_pairs": len(x),
              "optimizer_iterations": int(classifier.n_iter_[0]), "random_seed": 42,
              "input_records_hash": digest([r for r in records if r["split"] != "test"]),
              "validation_policy": "lowest threshold with >=90% precision on >=10 validation predictions",
              "label_source": "published_hyperlinks; project split; partially annotated",
              "model_type": "supervised logistic candidate ranker over pretrained E5 features"}
    write_json(destination / "model.json", config)
    metrics = {}
    for split in (("validation", "test") if evaluate_test else ("validation",)):
        output = validation if split == "validation" else predictions(split)
        for p in output:
            p["accepted_qid"] = p["predicted_qid"] if p["score"] >= threshold else None
        accepted = [p for p in output if p["accepted_qid"]]
        metrics[split] = {"mentions": len(output), "accepted": len(accepted),
            "coverage": len(accepted) / max(1, len(output)),
            "accepted_precision": sum(p["accepted_qid"] == p["qid"] for p in accepted) / max(1, len(accepted)),
            "top1_accuracy": sum(p["predicted_qid"] == p["qid"] for p in output) / max(1, len(output)),
            "alias_accuracy": sum(p["alias_prediction"] == p["qid"] for p in output) / max(1, len(output)),
            "candidate_coverage": sum(p["gold_in_catalogue"] for p in output) / max(1, len(output))}
        write_jsonl(destination / f"{split}_predictions.jsonl", output)
    metrics["status"] = "trained" if threshold < 1 else "trained_but_validation_precision_gate_not_met"
    metrics["limitations"] = ["Candidate KB is a training-derived subset, not all Wikidata",
                              "Unknown mentions use abstention, not invented IDs",
                              "Logistic scores are not calibrated probabilities"]
    write_json(destination / "metrics.json", metrics)
    write_jsonl(destination / "training_mentions.jsonl", [item[0] for item in training])
    names = ["entities.json", "entity_vectors.npy", "model.json", "metrics.json",
             "training_mentions.jsonl", "validation_predictions.jsonl"]
    if evaluate_test:
        names.append("test_predictions.jsonl")
    write_json(destination / "training_complete.json", {"identity": training_identity,
        "files": {name: file_digest(destination / name) for name in names}})
    print(metrics, flush=True)
    return metrics
