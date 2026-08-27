# Legacy configs

Not needed to reproduce the paper. Two groups:

- `*_gemma.yaml` target `google/gemma-7b`, an earlier backbone replaced by
  Gemma-2-9B before the reported experiments.
- `*_gemma2_9b.yaml` are the pre-safe-schedule Gemma-2-9B drafts
  (lr `2e-4`, clip `1.0`, warmup `100`). Every reported Gemma-2-9B run used the
  safer schedule of Appendix A (lr `1e-4`, clip `0.5`, warmup `300`), so these
  files do not describe any reported run.

Use `configs/canonical_pool/` for every reported cell; the Gemma-2-9B files
there carry the `_saferetrain` suffix.
