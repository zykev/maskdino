# tools/

Utilities for the intraoral `caries` / `orth` workflows.

## Project tools (orth / caries)

| Script | What it does | Typical usage |
| --- | --- | --- |
| `build_tooth_guidance.py` | Runs SAM-HQ over the orth images to produce tooth-mask guidance (`tooth_mask` image subdir) used as an extra input channel / prompt. | `python tools/build_tooth_guidance.py --help` |
| `sam.py` | Shared SAM-HQ wrapper (predictor / mask helpers) imported by `build_tooth_guidance.py`. Not a CLI entry point. | — |
| `analyze_orth_geometry.py` | Box geometry statistics over the orth LabelMe annotations (box size, area, aspect-ratio histograms) at several resize scales, written to `reports/`. | `python tools/analyze_orth_geometry.py` |
| `analyze_orth_test_annotations.py` | Per-class montage + statistics of the orth rectangle annotations, written to `output/orth_annotation_analysis/`. | `python tools/analyze_orth_test_annotations.py --draw-other-labels` |
| `visualize_orth_augmentations.py` | Visualizes the orth input/augmentation pipeline as montage tiles. | `python tools/visualize_orth_augmentations.py` |

Useful flags for the analysis tools:

* `analyze_orth_geometry.py` — `--data-dir`, `--out-dir`, `--scales`, `--max-size`, `--bins`
* `analyze_orth_test_annotations.py` — `--data-dir`, `--out-dir`, `--tile-size`, `--cols`, `--max-tiles-per-page`, `--draw-other-labels`
* `visualize_orth_augmentations.py` — `--data-dir`, `--out-dir`, `--sample-ids`, `--seed`, `--tile-width`, `--tile-height`

Each defaults to the current orth layout
(`.datasets/intraoral_anno/orth_0616/orth_0616`) and a folder under `reports/`
or `output/`, so they can be run with no arguments.


