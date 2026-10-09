"""Run one inference step on tensors of the documented shape."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch

from mmabc.action import unpack
from mmabc.batch import synthetic_batch
from mmabc.models.mmabc import MMABCPolicy, load_model_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/model/mmabc_4b.yaml")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=None, help="Euler steps. Defaults to the config.")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    cfg = load_model_config(args.config, repo_root=str(REPO))
    backbone = REPO / str(cfg.backbone.path)
    if not (backbone / "config.json").is_file():
        raise SystemExit(f"missing {backbone}\nRun: bash scripts/download_pretrained.sh")
    if args.steps is not None:
        cfg.flow.num_inference_steps = int(args.steps)
    cfg.future.enabled = False

    policy = MMABCPolicy(cfg, repo_root=str(REPO)).to(device).eval()
    spec = policy.input_spec()
    batch = synthetic_batch(policy, args.batch_size, train=False, device=device, seed=0)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        context = torch.autocast("cuda", dtype=torch.bfloat16)
    else:
        context = torch.autocast("cpu", enabled=False)
    with context:
        action = policy.predict_action(batch)

    expected = (args.batch_size, spec["chunk_size"], spec["action_dim"])
    if tuple(action.shape) != expected:
        raise SystemExit(f"action shape {tuple(action.shape)} != {expected}")
    raw = unpack(action[0, : spec["execute_steps"]].float().cpu().numpy())
    print(f"pred_type={spec['pred_type']} loss_type={spec['loss_type']}")
    print(f"action {tuple(action.shape)}  execute {spec['execute_steps']}")
    print("keys " + ", ".join(raw))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
