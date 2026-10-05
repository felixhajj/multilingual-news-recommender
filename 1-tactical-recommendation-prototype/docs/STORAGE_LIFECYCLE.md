# Storage After the Portfolio Release

The additional article processing expands the recommendation collection. It does
not require downloading another large model. The current 5,000-article corpus is
23.01 MiB, and the database containing 91 selected-version plus legacy results is
14.48 MiB. The large local costs are model downloads, dependencies and checkpoints.

## What Remains Available

The hosted application keeps the final 300 processed articles, model-generated
facts, source attribution, entity links and vectors. Changing interests and
analyzing new text work on the server after the PC's local files are removed.
The final collection must have a checksum-verified off-PC backup.

A compact local release keeps source code, guides, executed notebooks and measured
evidence. The runtime artifact bundle targets less than 500 MiB and includes every
file needed to verify the locked model selection. A separate private evidence
archive retains the full datasets, reviews, attempts, predictions, manifests,
adapters, resumable checkpoints and recovery history.

After deleting local models and dependencies, running inference or training on
this PC again requires restoring artifacts, installing dependencies and downloading
the pinned Qwen/E5 versions. Reading saved results and using the public demo do not.

## Cleanup Gates

Cleanup starts only after Phase 5 passes: public inference works without the local
server; runtime restoration succeeds in isolation; the off-PC evidence archive is
downloaded and verified; representative checkpoints and evaluation files restore.

The default dry run lists exact removable paths, sizes, retained files and backup
checksums. Active workers, missing backups, changed files, protected dependencies
and paths outside the allowlist block removal. Shared model caches and Windows
paging files are outside automatic cleanup.

Measured project-owned candidates on October 1: `.hf-cache` 5.75 GiB, `.venv`
5.41 GiB, `.pip-cache` 2.33 GiB, plus about 1 GiB of archived intermediate
checkpoints. Approximately 14 GiB reclaimed is a target, not a guarantee; final
physical free-space measurements determine the result. No files have been deleted
as part of approving this storage lifecycle.
