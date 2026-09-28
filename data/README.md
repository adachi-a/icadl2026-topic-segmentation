# Released data

## Split

`dataset_split.json` lists 20 development programs and the disjoint 20-program evaluation set.
Only the evaluation annotations and predictions are included in this release.

## Ground truth

- `ground_truth/recording_time/`: the coordinates used for the main boundary metrics.
- `ground_truth/logical_time/`: the same Topics after removal of recording lead-in/lead-out.
- `ground_truth/end_boundaries/`: end-boundary direction and presentation annotations, with both
  recording-time and logical-time coordinates.

Annotator identifiers and free-form workflow notes are removed. Topic labels, IPTC domains,
boundaries, and structured boundary-type tags are retained.

## Frozen system output

`segmentations/` contains only times and segment identifiers. Together with `ground_truth/`, it is
the complete input to the main evaluation; the scored per-program values are in `results/`. `generated_metadata/` retains generated titles, summaries, domains, and
non-editorial intervals, but applies these boundary-preserving transformations:

1. replace each private source path with `media-not-distributed/<program_id>.ts`;
2. empty transcript/OCR-derived `evidence_texts`;
3. remove free-form Topic and non-editorial notes;
4. empty run-specific warnings and internal frame/scene identifier lists.

These changes do not alter any Topic start or end time and therefore do not alter the reported segmentation
metrics.

## Baselines

The baseline directory contains already-generated outputs only.

| Directory | Condition name in `results/` | Notes |
|---|---|---|
| `texttiling/` | `texttiling` | SudachiPy mode C; parameters in each file |
| `embedding_window/` | `embedding_window_minilm` | `paraphrase-multilingual-MiniLM-L12-v2` |
| `embedding_window_ruri_v3_30m/` | `embedding_window_ruri` | `cl-nagoya/ruri-v3-30m` |
| `chapter_llama_frames_asr/` | `chapter_llama` | Chapter-Llama |
| `chapter_llama_asr/` | `chapter_llama_captions_asr` | Chapter-Llama |

Chapter-Llama files map `HH:MM:SS` chapter start times to generated chapter titles.

Recomputing embedding or
Chapter-Llama predictions may require external model downloads, GPUs, and separate upstream code;
it is not required to inspect the frozen comparison.
