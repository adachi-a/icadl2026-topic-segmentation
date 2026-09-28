# Japanese broadcast-news topic segmentation: ICADL 2026 artifact

This repository contains the production pipeline, its prompts, the 20-program evaluation split,
human topic-boundary annotations, frozen predictions, and reported artifacts for the ICADL 2026
paper.

## Contents

- `scripts/`, `src/`, `schemas/`: the code required by the production pipeline.
- `prompts/`: the original Japanese prompts and English reference translations.
- `protocol/`: the frozen experimental conditions and annotation guidelines.
- `data/ground_truth/`: curated human annotations in recording and logical time coordinates.
- `data/segmentations/`: boundary-only frozen predictions for five primary conditions.
- `data/generated_metadata/`: generated metadata with local paths, evidence excerpts, internal notes, and warnings removed.
- `data/baseline_segmentations/`: frozen TextTiling, embedding, and Chapter-Llama outputs.
- `results/`: reported per-program and aggregate results, paired bootstrap intervals, and runtime summaries.

Raw broadcasts, extracted audio, subtitles, ASR transcripts, OCR text streams, representative
frames, VLM frame annotations, evidence windows, execution logs, and the paper PDF are not
distributed. See `EXCLUDED_DATA.md` and `data/README.md`.

## Running the production pipeline

The production entry point is `scripts/run_mvp_production_ts_metadata.py`. Running the pipeline
requires user-supplied broadcast media, FFmpeg,
ARIB caption extraction, ASR/OCR environments, and configured model credentials. It can incur
substantial API and compute costs.

```bash
uv sync
uv venv .venv-asr --python 3.12
uv pip install --python .venv-asr/bin/python -r requirements-asr-runtime.txt
uv venv .venv-ocr --python 3.12
uv pip install --python .venv-ocr/bin/python -r requirements-ocr-runtime.txt
cp .env.example .env   # then fill in the model credentials
```

The pipeline uses the Japanese prompt files to process Japanese broadcasts. Files with `_en` in
their names are faithful reference translations and were not used for the reported runs.

The caption wrapper expects the external `monyone/assdumper` project; that third-party source and
its fonts are not vendored.
Clone it into `assdumper/` at the repository root (`src/extract_captions.py` runs
`assdumper/assdumper.py` from that directory):

```bash
git clone https://github.com/monyone/assdumper.git assdumper
```

The command-line options for each experimental condition are listed under `conditions` in
`protocol/frozen_conditions_v2.json`.

## Protocol notes

- `protocol/frozen_conditions_v2.json` is the frozen protocol restricted to the conditions reported
  in the paper; file hashes, evaluation-tool invocations, and unreported baselines were removed,
  and the remaining fields are as frozen. Relative paths such as `output/production_v2/...` refer
  to the original experiment workspace and do not exist in this repository.
- The ruri-v3-30m embedding baseline (`embedding_window_ruri_v3_30m`) and the Chapter-Llama
  comparison are not defined in the frozen protocol. Their parameters are recorded in the
  released output files.
- The Chapter-Llama comparison is a system-level comparison, not a same-component ablation: the
  frozen predictions were produced from the original TS recordings, whereas Chapter-Llama was run
  on MP4 files verified to be aligned with them.

## License

Code and prompts are released under the MIT License (`LICENSE`). Annotations, frozen predictions,
results, and protocol files are released under CC BY 4.0 (`LICENSE-DATA.md`). Neither license
covers the underlying broadcasts.
