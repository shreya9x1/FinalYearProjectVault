from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import TrainConfig
from dataset_llvip import LLVIPPairedDataset
from models.bridge import MVGSBModel


def psnr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = torch.mean((pred - target) ** 2)
    return 20.0 * torch.log10(torch.tensor(1.0, device=pred.device)) - 10.0 * torch.log10(mse + 1e-8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    cfg = TrainConfig()
    ds = LLVIPPairedDataset(cfg.dataset_root, split="test", image_size=cfg.image_size)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=cfg.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MVGSBModel(sigma_max=cfg.sigma_max).to(device)

    ckpt = torch.load(Path(args.checkpoint), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    running_psnr = 0.0
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dl, desc="Eval"), start=1):
            rgb = batch["rgb"].to(device)
            ir = batch["ir"].to(device)
            t = torch.full((rgb.size(0),), 0.5, device=device)

            out = model(rgb=rgb, ir=ir, t=t, semantic_scale=0.0, task_scale=0.0)
            running_psnr += float(psnr(out["pred_ir"], ir))

    print(f"Avg PSNR: {running_psnr / max(len(dl), 1):.3f}")


if __name__ == "__main__":
    main()
