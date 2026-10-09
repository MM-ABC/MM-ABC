"""Train for a few steps on synthetic batches of the documented shape."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch

from mmabc.batch import synthetic_batch
from mmabc.models.mmabc import MMABCPolicy, load_model_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/model/mmabc_4b.yaml")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--draws", type=int, default=1)
    parser.add_argument("--lr-backbone", type=float, default=1e-5)
    parser.add_argument("--lr-expert", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this training entry")
    device = torch.device(args.device)
    cfg = load_model_config(args.config, repo_root=str(REPO))
    cfg.flow.repeated_noise_draws = int(args.draws)
    backbone = REPO / str(cfg.backbone.path)
    teacher = REPO / str(cfg.future.vggt.ckpt_path)
    teacher_code = REPO / str(cfg.future.vggt.repo_path) / "vggt_omega"
    if not (backbone / "config.json").is_file() or not teacher_code.is_dir() or not any(teacher.glob("*.pt")):
        raise SystemExit("missing pretrained weights\nRun: bash scripts/download_pretrained.sh")

    policy = MMABCPolicy(cfg, repo_root=str(REPO)).to(device).train()
    spec = policy.input_spec()
    opt_cfg = type("Opt", (), {"lr_backbone": args.lr_backbone, "lr_expert": args.lr_expert, "weight_decay": 1e-8})()
    optim = torch.optim.AdamW(policy.param_groups(opt_cfg), betas=(0.9, 0.95), eps=1e-8)
    torch.backends.cuda.matmul.allow_tf32 = True

    for step in range(args.steps):
        batch = synthetic_batch(policy, args.batch_size, train=True, device=device, seed=step)
        optim.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = policy(batch)
        if not torch.isfinite(out.loss):
            raise SystemExit(f"step {step}: non-finite loss {out.metrics}")
        out.loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        if not torch.isfinite(grad):
            raise SystemExit(f"step {step}: non-finite gradient")
        optim.step()
        print(
            f"step {step} loss={out.metrics['loss']:.4f} "
            f"action={out.metrics['loss_action']:.4f} "
            f"future={out.metrics.get('loss_future', 0.0):.4f} "
            f"grad={float(grad):.3f} "
            f"chunk={spec['chunk_size']} dim={spec['action_dim']} "
            f"pred={spec['pred_type']} loss_type={spec['loss_type']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
