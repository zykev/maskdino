"""Dump the structure of the SSL pretrain checkpoints to verify key names/shapes."""

import argparse
import argparse as _argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

torch.serialization.add_safe_globals([_argparse.Namespace])

from maskdino.modeling.backbone.ssl_common import (  # noqa: E402
    CANDIDATE_STATE_KEYS,
    extract_state_dict,
)


def describe(checkpoint, label):
    print(f"\n===== {label} =====")
    if isinstance(checkpoint, dict):
        print("top-level:", list(checkpoint.keys())[:20])
    state, key = extract_state_dict(checkpoint, "")
    print(f"auto key={key!r}  tensors={len(state)}")
    keys = list(state.keys())
    print("first 15:", keys[:15])
    print("last 10:", keys[-10:])

    # Swin shape fingerprint
    for probe in ("patch_embed.proj.weight", "patch_embed.proj.bias"):
        if probe in state:
            print(f"  {probe}: {tuple(state[probe].shape)}")
    for stage in range(4):
        dims = {
            int(k.split(".")[1])
            for k in keys
            if k.startswith("layers.") and len(k.split(".")) > 2 and k.split(".")[1].isdigit()
        }
        pass
    stage_counts = {}
    for k in keys:
        parts = k.split(".")
        if parts[0] == "layers" and len(parts) > 1 and parts[1].isdigit():
            stage_counts.setdefault(int(parts[1]), set()).add(parts[2])
    for stage in sorted(stage_counts):
        print(f"  layers.{stage}: {len(stage_counts[stage])} blocks")
    for stage in sorted(stage_counts):
        probe = f"layers.{stage}.blocks.0.attn.qkv.weight"
        if probe in state:
            print(f"  {probe}: {tuple(state[probe].shape)}")

    # positions / cls tokens / registers / fc
    for probe in ("pos_embed", "cls_token", "register_tokens", "mask_token", "fc.weight",
                  "norm.weight", "norm1.weight"):
        if probe in state:
            print(f"  {probe}: {tuple(state[probe].shape)}")
    for probe in ("net.0.0.weight", "net.0.1.weight", "net.1.0.conv1.weight", "net.4.2.conv3.weight",
                  "0.0.weight", "1.0.conv1.weight"):
        if probe in state:
            print(f"  {probe}: {tuple(state[probe].shape)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()
    for path in args.paths:
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:  # noqa: BLE001
            print(f"{path}: FAILED to load: {exc}")
            continue
        describe(checkpoint, os.path.basename(path))


if __name__ == "__main__":
    main()
