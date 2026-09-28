# Excluded data

The following materials were used or produced in the experiments but are not distributed in this
repository.

| Material | Reason |
|---|---|
| Raw broadcast recordings (MPEG-TS) | Copyrighted broadcast content |
| Extracted audio | Derived from copyrighted broadcasts |
| ARIB subtitles and extracted caption files | Verbatim broadcast text |
| ASR transcripts and unified (caption/ASR fusion) transcripts | Verbatim broadcast speech |
| OCR text streams | Verbatim on-screen broadcast text |
| Representative frames | Broadcast images |
| VLM frame annotations | Descriptions of individual broadcast frames |
| Evidence windows and `evidence_texts` | Excerpts of transcript and OCR text |
| Execution logs and intermediate run directories | Contain local paths and broadcast-derived text |
| Development-set annotations and predictions | Only the evaluation set is released (`data/dataset_split.json`) |
| Paper PDF | Distributed by the publisher |

Because the raw media are not distributed, the pipeline cannot be re-run on the same inputs from
this repository alone. The frozen predictions in `data/segmentations/` and
`data/baseline_segmentations/`, together with `data/ground_truth/`, are the inputs to every
reported segmentation metric.

The transformations applied to released generated metadata are described in `data/README.md`.
