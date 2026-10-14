# Legacy flat configs (retired)

These files are the original *flat* Mask R-CNN / MaskDINO configs that used to
sit directly in `configs/`. They have been superseded by the organised tree:

```
configs/maskrcnn/{caries,orth}/{_base,no_ssl,simclr,swinmae,dentmag_swin,dinov2}.yaml
configs/maskdino/{caries,orth}/{_base,no_ssl,simclr,swinmae,dentmag_swin,dinov2}.yaml
```

They are kept only so that old experiment records (`output/*/config.yaml`,
W&B runs, shell history) remain interpretable. **Do not use them for new runs.**
Nothing in the repository references them any more; the entry points
(`maskrcnn_unify.py`, `maskdino_unify.py`) and everything in `scripts/` now
default to `configs/<detector>/<task>/no_ssl.yaml`.

## Mapping from legacy to the new tree

| Legacy file | Replacement |
| --- | --- |
| `default_maskrcnn_caries_config.yaml` | `configs/maskrcnn/caries/_base.yaml` + `no_ssl.yaml` |
| `default_maskrcnn_orth_config.yaml` | `configs/maskrcnn/orth/_base.yaml` + `no_ssl.yaml` |
| `default_maskrcnn_orth_swin_scratch_config.yaml` | `configs/maskrcnn/orth/swinmae.yaml` with `MODEL.BACKBONE.USE_SSL_PRETRAINED: false` |
| `default_maskrcnn_orth_swin_mae_config.yaml` | `configs/maskrcnn/orth/swinmae.yaml` |
| `default_maskdino_caries_config.yaml` | `configs/maskdino/caries/_base.yaml` + `no_ssl.yaml` |
| `default_maskdino_orth_base_config.yaml` | `configs/maskdino/orth/_base.yaml` |
| `default_maskdino_orth_resnet_config.yaml` | `configs/maskdino/orth/no_ssl.yaml` |
| `default_maskdino_orth_swin_scratch_config.yaml` | `configs/maskdino/orth/swinmae.yaml` with `MODEL.BACKBONE.USE_SSL_PRETRAINED: false` |
| `default_maskdino_orth_swin_mae_config.yaml` | `configs/maskdino/orth/swinmae.yaml` |
| `default_maskrcnn_config.yaml` | unused; a general COCO Mask R-CNN baseline |

## A from-scratch backbone is a variant, not a separate directory

The old `*_swin_scratch` configs are expressed in the new tree by taking the
matching SSL variant and turning the weights off:

```yaml
MODEL:
  BACKBONE:
    USE_SSL_PRETRAINED: false
    SSL_WEIGHTS: ""
```

The architecture (depth widths, patch size, window size, position encoding,
FFN ratio, ...) stays identical, so the only difference from the pretrained run
is the initialisation — which is exactly what a scratch control is for.

## Stale values fixed on the way out

While these files were still live they carried two bugs that are now corrected
here as well, so that old commands still resolve:

* `MODEL.SWIN.PRETRAINED_WEIGHTS` pointed at
  `/home/zychen/Documents/intraoral_code/Swin-MAE/exp/swinmae_v1/checkpoint-400.pth`,
  which does not exist in this checkout → now
  `.checkpoints/pretrain/swinmae/swins_checkpoint-50.pth`.
* `MODEL.SWIN.DEPTHS` was `[2, 2, 2, 2]` → now `[2, 2, 18, 2]`, the Swin-Small
  layout the checkpoints were trained with.
