"""Identity-bound Phase 3 validation, fixed decisions, then honest final reporting."""
import argparse
import inspect
import json
import sys
import sqlite3
import subprocess
import traceback
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.finish_phase2 import verify_completion
from scripts.validate_phase3_linker import metrics as linker_metrics, VALIDATION_POLICY
from src.learned_linker import (LearnedEntityLinker, encode_batch, normalized,
                                linker_artifact_identity, train_linker, context_at)
from src.news_evaluation import ranking_metrics, validate_ranking_review
from src.extraction_data import categories_for_schema
from src.llm_extractor import parse_first_json_object, validate_extraction_response
from src.news_pipeline import rank_records
from src.news_selection import lock_selection, verify_selection_lock, score_extraction, extraction_inputs
from src.portfolio_config import (ARTIFACTS, EMBEDDING_MODEL, EMBEDDING_REVISION, GPU_LOCK,
                                  digest, file_digest, read_json, read_jsonl, write_json, write_jsonl)
from src.phase3_resources import check_storage
from src.checkpoint_loading import checkpoint_loading

PHASE = ARTIFACTS / "phase3"
LINKER = PHASE / "linker-v3"
RANKING = PHASE / "ranking"
POLICY = {"linker": VALIDATION_POLICY, "ranking": {
    "semantic_weight": .75, "selection": "hybrid if validation Recall@5 and nDCG@5 are not below keyword; otherwise E5 under the same rule; otherwise keyword",
    "scope": "Fixed legacy-extractor index, same ten judged articles per query; not a metric for new Qwen adapters or corpus-wide recall"},
    "extraction": "Existing v3 gate unchanged; validation only before lock",
    "final_linker_scope": "Reuse the previously reported article-disjoint holdout; not a fresh test set"}


def immutable(path, payload):
    old = read_json(path)
    if old is not None and old != payload:
        raise ValueError(f"Immutable evidence changed: {path.name}")
    write_json(path, payload)
    return payload


def policy():
    return immutable(PHASE / "validation_policy.json", POLICY)


def train_and_validate_linker():
    policy()
    if not (LINKER / "inference_manifest.json").exists():
        if (LINKER.exists() and any(LINKER.iterdir())
                and not (LINKER / "training_inputs.json").exists()):
            raise ValueError("Partial linker preparation retained; inspect it before repeating training")
        records = read_jsonl(ARTIFACTS / "corpus.jsonl")
        expected = []
        for article in sorted(records, key=lambda r: digest(r["article_id"])):
            if article["split"] == "validation":
                for mention in article["mentions"][:5]:
                    if len(expected) < 300:
                        expected.append((article["article_id"], mention["start"], mention["end"], mention["text"], mention["qid"]))
        original = read_jsonl(ARTIFACTS / "linker/validation_predictions.jsonl")
        if expected != [(r["article_id"], r["start"], r["end"], r["text"], r["qid"]) for r in original]:
            raise ValueError("Frozen linker validation pool differs; refuse unnecessary training")
        # The only supervised targets are published mention/QID positives from training articles.
        with checkpoint_loading():
            train_linker(records, 2000, 300, directory=LINKER, evaluate_test=False)
        identity = linker_artifact_identity(LINKER)
        immutable(LINKER / "inference_manifest.json", {
            "artifact_identity": identity,
            "validation_predictions_sha256": file_digest(LINKER / "validation_predictions.jsonl"),
            "training_mentions_sha256": file_digest(LINKER / "training_mentions.jsonl"),
            "training_implementation": digest(inspect.getsource(train_linker)),
            "corpus_sha256": file_digest(ARTIFACTS / "corpus.jsonl"),
            "annotation_scope": "Published hyperlink positives, not exhaustive human-gold entities",
            "test_predictions_generated": False})
    identity = linker_artifact_identity(LINKER)
    manifest = read_json(LINKER / "inference_manifest.json")
    if (manifest["artifact_identity"] != identity
            or file_digest(LINKER / "validation_predictions.jsonl") != manifest["validation_predictions_sha256"]):
        raise ValueError("Linker validation evidence changed")
    rows = read_jsonl(LINKER / "validation_predictions.jsonl")
    original = read_jsonl(ARTIFACTS / "linker/validation_predictions.jsonl")
    keys = lambda entries: [(r["article_id"], r["start"], r["end"], r["text"], r["qid"]) for r in entries]
    if keys(rows) != keys(original):
        raise ValueError("Linker validation pool changed")
    config, entities = read_json(LINKER / "model.json"), read_json(LINKER / "entities.json")
    result = linker_metrics(rows, entities, config["threshold"])
    passed = (result["accepted"] >= 10 and (result["accepted_precision"] or 0) >= .9
              and (result["unknown_abstention"] if result["unknown_abstention"] is not None else 1) >= .9
              and result["correct_link_accuracy"] > result["unique_alias_accuracy"])
    return immutable(PHASE / "reports/linker_v3_validation.json", {
        "role": "validation", "metrics": result, "gate_passed": passed,
        "by_language": {lang: linker_metrics([r for r in rows if r["language"] == lang], entities, config["threshold"])
                        for lang in ("en", "ar")},
        "policy": POLICY["linker"], "inference_manifest_sha256": file_digest(LINKER / "inference_manifest.json"),
        "artifact_identity": identity, "label_source": "published hyperlink mention/QID annotations; partially annotated"})


def prepare_ranking():
    policy()
    judgments = read_jsonl(ARTIFACTS / "review/ranking.jsonl")
    if validate_ranking_review(judgments) != 30:
        raise ValueError("Frozen recommendation review is incomplete")
    path = RANKING / "manifest.json"
    if not path.exists():
        release = read_json(ARTIFACTS / "release_manifest.json")
        expected_encoder = f"{EMBEDDING_MODEL}:{EMBEDDING_REVISION}:normalized-chunk-mean-v2"
        if release["embedding_version"] != expected_encoder:
            raise ValueError("Legacy index encoder differs from the pinned query encoder")
        wanted = {r["article"]["article_id"]: r["article"] for r in judgments}
        uri = (ARTIFACTS / "release.sqlite").resolve().as_uri() + "?mode=ro"
        with sqlite3.connect(uri, uri=True) as db:
            stored = db.execute("SELECT article_id, payload, embedding_version, vector FROM articles ORDER BY article_id").fetchall()
        records, vectors = [], []
        for article_id, payload, encoder, vector in stored:
            if article_id not in wanted:
                continue
            record = json.loads(payload)
            if (record["model"] != release["model"] or record["pipeline_identity"] != release["pipeline_identity"]
                    or encoder != release["embedding_version"] or record["status"] != "ready"
                    or any(record["article"][k] != wanted[article_id][k] for k in ("title", "body"))):
                raise ValueError("Frozen ranking article differs from its stored inference/index")
            records.append(record)
            vectors.append(np.frombuffer(vector, dtype=np.float32).copy())
        if {r["article"]["article_id"] for r in records} != set(wanted):
            raise ValueError("Missing judged articles; never substitute other articles")
        queries = {r["query_id"]: "query: " + r["interest"].strip() for r in judgments}
        query_ids = sorted(queries)
        with checkpoint_loading():
            query_vectors = encode_batch([queries[k] for k in query_ids])
        RANKING.mkdir(parents=True, exist_ok=True)
        write_json(RANKING / "records.json", records)
        np.save(RANKING / "article_vectors.npy", np.stack(vectors), allow_pickle=False)
        np.save(RANKING / "query_vectors.npy", query_vectors, allow_pickle=False)
        immutable(path, {"query_ids": query_ids, "queries": queries,
            "files": {name: file_digest(RANKING / name) for name in ("records.json", "article_vectors.npy", "query_vectors.npy")},
            "reference_sha256": file_digest(ARTIFACTS / "review/ranking.jsonl"),
            "release_manifest_sha256": file_digest(ARTIFACTS / "release_manifest.json"),
            "indexed_model": release["model"], "indexed_embedding": release["embedding_version"],
            "query_encoder": EMBEDDING_MODEL, "query_encoder_revision": EMBEDDING_REVISION,
            "ranking_implementation": digest(inspect.getsource(rank_records)),
            "scope": POLICY["ranking"]["scope"]})
    return ranking_report("validation")


def ranking_report(role):
    if role == "test":
        verify_selection_lock()
    manifest = read_json(RANKING / "manifest.json")
    for name, sha in manifest["files"].items():
        if file_digest(RANKING / name) != sha:
            raise ValueError("Frozen ranking vectors/records changed")
    if (manifest["reference_sha256"] != file_digest(ARTIFACTS / "review/ranking.jsonl")
            or manifest["ranking_implementation"] != digest(inspect.getsource(rank_records))):
        raise ValueError("Ranking references or shared implementation changed")
    rows = [r for r in read_jsonl(ARTIFACTS / "review/ranking.jsonl") if r["role"] == role]
    records = read_json(RANKING / "records.json")
    vectors = np.load(RANKING / "article_vectors.npy", allow_pickle=False)
    queries = np.load(RANKING / "query_vectors.npy", allow_pickle=False)
    output = {}
    for query_id in sorted({r["query_id"] for r in rows}):
        pool = [r for r in rows if r["query_id"] == query_id]
        relevance = {r["article"]["article_id"]: r["relevance"] for r in pool}
        indices = [i for i, r in enumerate(records) if r["article"]["article_id"] in relevance]
        if len(indices) != 10 or len(relevance) != 10:
            raise ValueError("Expected the unchanged ten-article judged pool")
        query = pool[0]
        vector = queries[manifest["query_ids"].index(query_id)]
        output[query_id] = {}
        for method in ("keyword", "e5", "hybrid"):
            ranked = rank_records([records[i] for i in indices], vectors[indices], query["interest"],
                                  query["required_filters"], method, .75, vector)
            ids = [r["article"]["article_id"] for r in ranked]
            output[query_id][method] = {"ranked_ids": ids, "metrics": ranking_metrics(ids, relevance)}
    averages = {method: {key: float(np.mean([q[method]["metrics"][key] for q in output.values()]))
                         for key in ("recall_at_5", "ndcg_at_5")} for method in ("keyword", "e5", "hybrid")}
    return immutable(PHASE / "reports" / f"ranking_{role}.json", {
        "role": role, "queries": output, "averages": averages, "policy": POLICY["ranking"],
        "input_manifest_sha256": file_digest(RANKING / "manifest.json"),
        "scope": manifest["scope"], "judgment_source": "human_reviewed"})


def lock_decisions():
    policy()
    linker = read_json(PHASE / "reports/linker_v3_validation.json")
    ranking = read_json(PHASE / "reports/ranking_validation.json")
    scores = ranking["averages"]
    selected_method = "keyword"
    for method in ("hybrid", "e5"):
        if all(scores[method][key] >= scores["keyword"][key] for key in ("recall_at_5", "ndcg_at_5")):
            selected_method = method
            break
    immutable(PHASE / "component_decisions.json", {
        "linker_selected": "phase3/linker-v3" if linker["gate_passed"] else None,
        "ranking_method": selected_method, "semantic_weight": .75,
        "ranking_scope": POLICY["ranking"]["scope"], "policy": POLICY})
    paths = [PHASE / "validation_policy.json", PHASE / "component_decisions.json",
             PHASE / "reports/linker_v3_validation.json", PHASE / "reports/ranking_validation.json",
             LINKER / "inference_manifest.json", LINKER / "model.json", LINKER / "entities.json",
             LINKER / "entity_vectors.npy", LINKER / "validation_predictions.jsonl",
             RANKING / "manifest.json", RANKING / "records.json", RANKING / "article_vectors.npy", RANKING / "query_vectors.npy"]
    evidence = {p.relative_to(ARTIFACTS).as_posix(): file_digest(p) for p in paths}
    for run in ("base-v3", "extraction-only-v3", "domain-extraction-v3"):
        validation_diagnostics(run)
        path = PHASE / "reports" / f"{run}_validation_errors.json"
        evidence[path.relative_to(ARTIFACTS).as_posix()] = file_digest(path)
    lock = lock_selection("base-v3", ["extraction-only-v3", "domain-extraction-v3"], evidence)
    for run in lock["run_evidence"]:
        score_extraction(run, "test")
    ranking_report("test")
    return lock


def validation_diagnostics(run_id):
    """Only validation references inform this error audit; test output stays unscored."""
    references, roles, predictions, schema, evidence = extraction_inputs(run_id)
    gold = {r["article_id"]: r for r in references if roles[r["article_id"]] == "validation"}
    categories = [c for c in categories_for_schema(schema) if c != "topics"]
    rows = []
    for prediction in predictions:
        if prediction["article_id"] not in gold:
            continue
        parsed, parse_error, schema_error = {}, None, None
        try:
            chunks = [parse_first_json_object(text) for text in prediction["raw_outputs"]]
            if not chunks:
                raise ValueError("Missing model output")
        except ValueError as exc:
            parse_error = str(exc)
            chunks = []
        if chunks:
            try:
                for chunk in chunks:
                    validate_extraction_response(chunk, schema)
                parsed = {c: {v.casefold().strip() for chunk in chunks for v in chunk[c]} for c in categories}
            except ValueError as exc:
                schema_error = str(exc)
        reference = gold[prediction["article_id"]]
        differences = {}
        for category in categories:
            target = {v.casefold().strip() for v in reference["labels"][category]}
            actual = parsed.get(category, set())
            differences[category] = {"missing": sorted(target - actual), "extra": sorted(actual - target),
                                     "correct": sorted(actual & target)}
        rows.append({"article_id": prediction["article_id"], "language": reference["language"],
                     "parse_error": parse_error, "schema_error": schema_error,
                     "inference_error": prediction["error"], "entity_differences": differences,
                     "chunk_metadata": prediction["chunk_metadata"]})
    return immutable(PHASE / "reports" / f"{run_id}_validation_errors.json", {
        "role": "validation", "run_id": run_id, "evidence": evidence, "examples": rows,
        "interpretation": "Exact normalized surface errors; schema failures count all gold entities as missing. Topics/relationships are not entity-F1 targets."})


def linker_final():
    verify_selection_lock()
    linker = LearnedEntityLinker(LINKER)
    source = ARTIFACTS / "linker/final_holdout.jsonl"
    gold = read_jsonl(source)
    corpus = {r["article_id"]: r for r in read_jsonl(ARTIFACTS / "corpus.jsonl")}
    for row in gold:
        article = corpus[row["article_id"]]
        if article["split"] != "test" or row["context"] != context_at(article["body"], row["start"], row["end"]):
            raise ValueError("Final linker input is not the preserved test-only article context")
    immutable(LINKER / "final_manifest.json", {
        "input_sha256": file_digest(source), "linker_identity": linker.artifact_identity,
        "scope": POLICY["final_linker_scope"], "selection_lock_sha256": file_digest(PHASE / "selection_lock.json")})
    path = LINKER / "final_predictions.jsonl"
    rows = read_jsonl(path) if path.exists() else []
    keys = lambda entries: [(r["article_id"], r["start"], r["end"], r["text"], r["qid"], r["context"]) for r in entries]
    if len(rows) > len(gold) or keys(rows) != keys(gold[:len(rows)]):
        raise ValueError("Final linker prediction cursor changed")
    qids = {e["qid"] for e in linker.entities}
    aliases = {}
    for entity in linker.entities:
        for alias in entity["normalized_aliases"]:
            aliases.setdefault(alias, set()).add(entity["qid"])
    with checkpoint_loading():
        for row in gold[len(rows):]:
            check_storage(ARTIFACTS)
            result = linker.link(row["text"], row["context"], candidate_limit=10)
            match = aliases.get(normalized(row["text"]), set())
            rows.append({**row, "predicted_qid": result["candidates"][0]["entity_id"],
                "score": result["unrounded_link_score"], "accepted_qid": result["entity_id"],
                "gold_in_catalogue": row["qid"] in qids,
                "gold_retrieved": row["qid"] in {c["entity_id"] for c in result["candidates"]},
                "alias_prediction": next(iter(match)) if len(match) == 1 else None,
                "prediction": result})
            write_jsonl(path, rows)
    config = read_json(LINKER / "model.json")
    return immutable(PHASE / "reports/linker_v3_test.json", {
        "role": "test", "metrics": linker_metrics(rows, linker.entities, config["threshold"]),
        "by_language": {lang: linker_metrics([r for r in rows if r["language"] == lang], linker.entities, config["threshold"])
                        for lang in ("en", "ar")},
        "manifest_sha256": file_digest(LINKER / "final_manifest.json"),
        "predictions_sha256": file_digest(path), "scope": POLICY["final_linker_scope"]})


def finish():
    lock = verify_selection_lock()
    decisions = read_json(PHASE / "component_decisions.json")
    for run in lock["run_evidence"]:
        if read_json(PHASE / "reports" / f"{run}_test.json") != score_extraction(run, "test"):
            raise ValueError("Final extraction report cannot be reproduced")
    ranking_report("test")
    final_linker = read_json(PHASE / "reports/linker_v3_test.json")
    if not final_linker or final_linker["predictions_sha256"] != file_digest(LINKER / "final_predictions.jsonl"):
        raise ValueError("Final learned-linker evidence missing or changed")
    rows = read_jsonl(LINKER / "final_predictions.jsonl")
    config, entities = read_json(LINKER / "model.json"), read_json(LINKER / "entities.json")
    gold = read_jsonl(ARTIFACTS / "linker/final_holdout.jsonl")
    keys = lambda entries: [(r["article_id"], r["start"], r["end"], r["text"], r["qid"], r["context"]) for r in entries]
    if keys(rows) != keys(gold):
        raise ValueError("Final linker predictions are not the complete frozen pool")
    expected_languages = {lang: linker_metrics([r for r in rows if r["language"] == lang], entities, config["threshold"])
                          for lang in ("en", "ar")}
    if (final_linker["metrics"] != linker_metrics(rows, entities, config["threshold"])
            or final_linker["by_language"] != expected_languages):
        raise ValueError("Final linker metrics cannot be reproduced")
    # Also covers controllers launched before the explicit verify stage was added.
    if not (PHASE / "completion.json").exists():
        verify_tests()
    test_proof = read_json(PHASE / "tests.json", {})
    if test_proof.get("passed") is not True:
        raise ValueError("The final test suite has not passed")
    paths = [p for p in (PHASE / "reports").glob("*_test.json")]
    ready = lock["selected_run"] is not None and decisions["linker_selected"] is not None
    report = {"phase": 3, "status": "evaluated" if ready else "evaluated_with_unmet_release_gates",
        "selection_lock_hash": lock["lock_hash"], "selected_extractor": lock["selected_run"],
        "component_decisions": decisions, "release_eligible": ready,
        "final_reports": {p.relative_to(ARTIFACTS).as_posix(): file_digest(p) for p in paths},
        "tests_sha256": file_digest(PHASE / "tests.json"),
        "budget": read_json(ARTIFACTS / "processing_budget.json"),
        "limitations": ["30 weak training examples; only 10 human-reviewed extraction validation articles.",
                        "One validation recommendation query and two test queries, ten judged articles each.",
                        "Retrieval comparison uses fixed legacy extraction metadata, not the newly selected adapter.",
                        "Linker final pool was previously reported and is not a fresh independent test.",
                        "Phase 4 must apply the validated configuration and rebuild affected indexes consistently."]}
    completed = immutable(PHASE / "completion.json", report)
    write_summary(completed)
    return completed


def write_summary(completed):
    lines = ["# Phase 3 Measured Results", "", "Generated from hash-bound saved predictions and references.", "",
             f"Evaluation status: **{completed['status']}**. Release eligible: **{completed['release_eligible']}**.", "",
             f"Selected extractor: `{completed['selected_extractor'] or 'none: validation gate not met'}`.",
             f"Ranking method: `{completed['component_decisions']['ranking_method']}`. No application model was automatically promoted.", "",
             "## Extraction", "", "The same frozen 10 validation and 20 test articles are used for all runs. Decisions use validation only.", "",
             "| Run | Role | JSON Parse | Schema | Precision | Recall | Entity F1 |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for run in ("base-v3", "extraction-only-v3", "domain-extraction-v3"):
        for role in ("validation", "test"):
            report = read_json(PHASE / "reports" / f"{run}_{role}.json")
            values = [f"{report['metrics'][key]:.3f}" for key in
                      ("json_parse_validity", "schema_validity", "precision", "recall", "micro_f1")]
            lines.append(f"| {run} | {role} | " + " | ".join(values) + " |")
    lines.extend(["", "Exact surface matching is used. Topics and relationships are schema-checked but not entity-F1 targets.", "",
                  "## Actual Training", "", "30 machine-assisted examples, not human-gold labels; all 160 labeling attempts remain preserved.", ""])
    for run in ("extraction-only-v3", "domain-extraction-v3"):
        training = read_json(ARTIFACTS / "runs" / run / "manifest.json")
        lines.append(f"- `{run}`: {training['optimizer_steps']} optimizer updates, {training['unique_articles_seen']} unique articles, "
                     f"{training['supervised_tokens']:,} supervised tokens, {training['changed_tensors']}/{training['total_trainable_tensors']} changed adapter tensors.")
    lines.extend(["", "Domain initialization reuses the completed bounded `domain-v1` experiment; raw-domain training was not repeated.", "",
                  "## Learned Linker", "", "E5 candidate retrieval plus a trained logistic ranker; catalogue entries are not manual decisions.", "",
                  "| Role | Accepted Precision | Coverage | Correct-Link Accuracy | Alias Baseline | Unknown Abstention |", "| --- | --- | --- | --- | --- | --- |"])
    for role in ("validation", "test"):
        metrics = read_json(PHASE / "reports" / f"linker_v3_{role}.json")["metrics"]
        values = ["N/A" if metrics[key] is None else f"{metrics[key]:.3f}" for key in
                  ("accepted_precision", "accepted_coverage", "correct_link_accuracy", "unique_alias_accuracy", "unknown_abstention")]
        lines.append(f"| {role} | " + " | ".join(values) + " |")
    lines.extend(["", "References are published hyperlink positives, not exhaustive human entity labels. The final linker pool was previously reported, not a fresh independent holdout.", "",
                  "## Recommendation Comparison", "", "Identical human-judged pools for keyword, E5 and hybrid; fixed legacy extraction metadata.", "",
                  "| Role | Method | Recall@5 | nDCG@5 |", "| --- | --- | --- | --- |"])
    for role in ("validation", "test"):
        scores = read_json(PHASE / "reports" / f"ranking_{role}.json")["averages"]
        for method in ("keyword", "e5", "hybrid"):
            lines.append(f"| {role} | {method} | {scores[method]['recall_at_5']:.3f} | {scores[method]['ndcg_at_5']:.3f} |")
    budget = completed["budget"]
    lines.extend(["", "## Limits and Next Phase", "", *[f"- {item}" for item in completed["limitations"]], "",
                  f"Charged cumulative processing allowance: {budget['consumed_seconds']/3600:.2f}/24 hours; "
                  f"{(budget['limit_seconds']-budget['consumed_seconds'])/3600:.2f} hours remain. Prepaid allowances are not measured runtimes.", "",
                  "Reproduce metrics with `python scripts/phase3_analysis.py finish`; no model inference is needed when saved evidence is intact.",
                  "Inspect validation failures in `output/portfolio/phase3/reports/*_validation_errors.json`.",
                  "Phase 4 applies the validated configuration only if gates pass, rebuilds affected indexes, and finishes the shared application/notebook journey.", ""])
    path = Path(__file__).resolve().parents[1] / "docs/PHASE_3_RESULTS.md"
    path.write_text("\n".join(lines), encoding="utf-8")


def verify_tests():
    result = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests"],
                            capture_output=True, text=True)
    print(result.stdout, result.stderr, flush=True)
    import re
    count = re.search(r"Ran (\d+) tests", result.stderr)
    proof = {"passed": result.returncode == 0, "tests": int(count.group(1)) if count else None,
             "output": result.stdout + result.stderr}
    write_json(PHASE / "tests.json", proof)
    if result.returncode:
        raise RuntimeError("Final test suite failed; no Phase 3 completion claim")
    return proof


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["policy", "linker", "ranking", "lock", "linker-test", "verify", "finish"])
    args = parser.parse_args()
    verify_completion()
    check_storage(ARTIFACTS)
    name = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-analysis-" + args.stage
    logs = PHASE / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    with (logs / (name + ".out.log")).open("w", encoding="utf-8") as out, \
         (logs / (name + ".err.log")).open("w", encoding="utf-8") as err, \
         redirect_stdout(out), redirect_stderr(err), GPU_LOCK.acquire(timeout=0):
        try:
            result = {"policy": policy, "linker": train_and_validate_linker, "ranking": prepare_ranking,
                      "lock": lock_decisions, "linker-test": linker_final, "verify": verify_tests,
                      "finish": finish}[args.stage]()
            print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        except BaseException:
            traceback.print_exc(file=err)
            raise
