# Results

- `main_conditions/`: per-program and aggregate evaluation outputs for the five primary conditions.
- `post_annotation/`: per-program start/end metrics and diagnostic aggregates used in the paper.
- `paired_bootstrap/`: paired, program-level percentile 95% confidence intervals (1,000 samples,
  seed 42).
- `baselines/`: TextTiling and embedding baseline evaluations (MiniLM and ruri-v3-30m), and
  `comparison_to_full/` against the Integrated (`full`) condition.
- `comparisons/`: paired bootstrap comparisons of `full` against the other primary conditions.
- `boundary_types/`: start-boundary recall by annotated boundary type for the `full` condition.
- `chapter_llama/`: the final frozen Chapter-Llama comparison. See `data/README.md` for the
  mapping between condition names and `data/baseline_segmentations/` directories.
- `runtime/`: observed elapsed time for the Integrated runs.

The runtime values come from runs that reused previously computed ASR files. They therefore
describe the recorded Integrated pipeline invocations in this experiment, not raw-media-to-output
latency with a fresh ASR pass.
