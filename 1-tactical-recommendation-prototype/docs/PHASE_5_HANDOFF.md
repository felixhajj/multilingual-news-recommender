# Phase 5 Handoff

October 5, 2026. Public functionality and private evidence restoration verified.
Final runtime publication/download/restoration passed; physical second-device
confirmation and temporary-copy housekeeping are pending. Phase 6 bulk cleanup
has not started and is prohibited until Phase 5 passes.

## Verified Work

- Runtime/evidence profiles, checksum manifests, frozen selection dependencies and compacted-database provenance are implemented. Runtime is approximately 123 MiB, below 500 MiB, excluding base models/dependencies.
- Clean Python 3.12 evidence dependencies installed from pinned requirements; pip check passed. Isolated offline replay reproduced 300 records, extraction F1 0.3134796, schema validity 0.95, and hybrid/E5 nDCG@5 0.9564838 versus keyword 0.6118293. Protected evidence remained unchanged. This is saved-metric replay, not fresh inference.
- All three default notebooks passed after relevant shared-app changes. Generation and training remain explicit opt-ins. 143 regression tests passed in 37.859 seconds; Gradio event-loop and oversized-token fixture warnings did not fail tests.
- Independent Astra High review closed source-level blockers, including form lock/unlock, stale-output clearing, mandatory acceptance assertions, ranged-download recovery and POSIX interpreter symlink preservation. See [audit](PHASE_5_AUDIT.md).
- Separate public code: [multilingual-news-recommender](https://github.com/felixhajj/multilingual-news-recommender). It does not inherit private history or publish large caches/credentials; the original repository remains private.
- Credentials remain in local service stores outside Git. No retraining, relabeling, selection changes or budget reset occurred.

## Public Proof And Backups

- Private full archive: `felixhajj/news-recommender-evidence`, immutable revision `48abc8c2be870cce91eb651687c16093515998eb`. Archive SHA-256 `4c828ba1f9a064ac9515756da6308cc18f28023172931fc520ed520ef05cc078`; 1,322,660,489 bytes. Remote download and all 661 member hashes passed. All 48 checkpoint members were verified; representative 288-tensor adapter plus optimizer/resume state, reviews and predictions were restored.
- [Public runtime](https://huggingface.co/datasets/felixhajj/multilingual-news-runtime): final immutable version, revision and checksum in `output/portfolio/phase5/runtime_upload.json`. Public `release.json` directs installers to the pinned archive; superseded candidates remain historical. Restore verifies all members.
  Final archive: `ee684f95d7582e11313fb86583bcabb5482e814167277b0346181665c5a25323`, revision `be5502991532fc86c698f6f7f41c4704f8ffad7a`, SHA-256 `0591865f8232a9f6092c49e1da61f6fffa881c30b40b9593234afdea276c6032`, 128,277,480 bytes. Immutable installer descriptor revision: `f3c8ec251284a824f94b0fbc5bafc21a6bf9e13e`. Remote SHA/member verification and metric replay passed; a separate clean dependency replay of this exact archive also passed.
- [Live Space](https://huggingface.co/spaces/felixhajj/multilingual-news-recommender): revision `e6fd21fad53d705a98b6f1151948c3dc45e2fa90`, free ZeroGPU running. Fresh synthetic English/Arabic inputs produced actual model traces, including observed mistakes. Edited input changed cache identity; unknown names remained unresolved. These are functionality checks, not quality evaluation.
- [Pages](https://felixhajj.github.io/multilingual-news-recommender/): visitor guide, saved evidence and links work without local model downloads. Public ranking changes with interests; result explanation, evidence and visible empty-interest handling passed.
- Desktop/mobile browser checks passed with local port 8502 stopped; no horizontal overflow at 1280/390 widths. Empty article analysis cleared old output, visibly failed and unlocked controls. A physical second-device check is still pending user confirmation.
- Hosted generation uses the same selected adapter with float16 base weights; frozen quality metrics use local 4-bit Qwen. Execution identities differ. Do not claim numerical/quality parity. Small evaluation pools, extraction mistakes and weak unseen-alias performance remain visible.
- Rebuilt public app matches local source SHA-256 `331bf0d31d0d0055d375cb6520e202a1d222e83803d9cf3c4a7e515b189fcb2f`. Actual quota rejection now returns empty failed output and unlocks controls. Fresh free-account English/Arabic and edited-input checks passed after this repair.
- Exploratory new-query failure: "European energy trade and natural gas" ranked a Minnesota oil pipeline first and an unrelated Google-slander story second. Some excerpts retain residual wikitext. These observations are recorded limitations, not new test labels or grounds for retuning the frozen model selection.
- Traces, screenshots and receipts: `output/portfolio/phase5/`; public summaries: `data/release/phase5_verification.json` and [dated acceptance](../data/release/phase5_acceptance.json). One partial Drive archive part is not a complete backup or cleanup gate.

## Next Steps

1. Receive the user's phone/other-laptop confirmation: open the portfolio, enter the live demo and change the interest. Do not count browser emulation as this check.
2. Resolve temporary-copy housekeeping. Final useful archives total about 1.35 GiB, but duplicate Phase 5 restores/downloads and packaging candidates total 8.56 GiB, exceeding the 5 GiB target when included. The automatic pruning attempt was rejected and no temporary/bulk cleanup was performed; specific duplicate-pruning approval was requested. Inventory: `phase5/storage_inventory.json`; D: had about 20.68 GiB free.
3. Only after Phase 5 passes, begin Phase 6's dry-run inventory and controlled bulk cleanup. Original corpus, environment, model caches and checkpoints remain protected.

The verified runtime manifest describes its exact build-time contents. Later
operational acceptance reports/documentation in Git do not alter that immutable
archive or its application/model version. The initial public source commit has one
historical 11 MiB tokenizer JSON (removed from current HEAD); no model weights,
environments, credential files or SQLite/ZIP artifacts were found in public history.

Budget remains 98,454.808/115,200 seconds (27.35/32 hours), approximately 4.65 hours left.
Network transfer is not model learning. Free GPU quotas/cold starts remain real constraints.
