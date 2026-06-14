
## starcoderdata shard (scoped 2026-06-13)
- HF path: `bigcode/starcoderdata` — GATED (must accept ToS to access; email+username shared w/ maintainers).
- License: "other". Per-file licenses of original source repos still apply (attribution/compliance required). Subject to validated data-removal requests (must use latest version).
- Subsets via `data_dir=`: 86 langs (e.g. `python`), `github-issues-filtered-structured`, `git-commits-cleaned`, `jupyter-scripts-dedup-filtered`, `jupyter-structured-clean-dedup`. NOTE: NO standalone "markdown" data_dir — markdown lives inside the per-language dirs (`markdown`/`restructuredtext`) as their own language folders; "issues" = github-issues subset.
- ~250B tok total (783GB code + 54GB issues + 13GB jupyter + 32GB commits).
- For [[Weaver]] coder-MoE pretrain: code-heavy (python/js/etc) + thin NL slice = markdown lang dir + github-issues. Stream w/ `streaming=True` (783GB won't fit disk).
- Mix weight ~0.18 of 6B-tok budget; modality=mixed (code-dominant).
