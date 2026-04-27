from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from skimage.feature import canny
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import TrainConfig, get_epoch_range_for_stage, get_stage_for_epoch
from dataset_llvip import LLVIPPairedDataset
from losses.objective import composite_loss
from models.bridge import MVGSBModel
from utils.schedules import curriculum_scale


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loader(cfg: TrainConfig, split: str) -> DataLoader:
    cache_dir = None
    if cfg.use_cached_dino_features:
        candidate = cfg.project_root / cfg.dino_cache_dir / split
        if candidate.exists():
            cache_dir = candidate

    ds = LLVIPPairedDataset(
        root=cfg.dataset_root,
        split=split,
        image_size=cfg.image_size,
        dino_cache_dir=cache_dir,
    )
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=(split == "train"),
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=(cfg.persistent_workers and cfg.num_workers > 0),
        prefetch_factor=(cfg.prefetch_factor if cfg.num_workers > 0 else None),
        drop_last=(split == "train"),
    )


def build_progress_samples(cfg: TrainConfig) -> list[dict[str, torch.Tensor | str]]:
    ds = LLVIPPairedDataset(root=cfg.dataset_root, split=cfg.progress_eval_split, image_size=cfg.image_size)
    n = min(cfg.progress_eval_num_samples, len(ds))
    samples: list[dict[str, torch.Tensor | str]] = []
    for i in range(n):
        item = ds[i]
        samples.append({"id": item["id"], "rgb": item["rgb"], "ir": item["ir"]})
    return samples


def try_build_lpips_model(use_lpips: bool, device: torch.device):
    if not use_lpips:
        return None
    try:
        import lpips  # type: ignore

        model = lpips.LPIPS(net="alex").to(device)
        model.eval()
        return model
    except Exception:
        return None


def to_rgb_uint8(x: torch.Tensor) -> np.ndarray:
    arr = x.detach().cpu().numpy()
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = np.repeat(arr, 3, axis=0)
    arr = np.transpose(arr, (1, 2, 0))
    arr = np.clip(arr, 0.0, 1.0)
    return (arr * 255.0).astype(np.uint8)


def normalize_gray_for_viz(x: torch.Tensor) -> np.ndarray:
    arr = x.detach().cpu().numpy().squeeze()
    mn = float(arr.min())
    mx = float(arr.max())
    arr = (arr - mn) / (mx - mn + 1e-8)
    arr = np.repeat(arr[..., None], 3, axis=2)
    return (arr * 255.0).astype(np.uint8)


def compute_canny_iou(pred_gray: np.ndarray, gt_gray: np.ndarray, sigma: float = 1.0) -> float:
    pred_e = canny(pred_gray, sigma=sigma)
    gt_e = canny(gt_gray, sigma=sigma)
    inter = np.logical_and(pred_e, gt_e).sum()
    union = np.logical_or(pred_e, gt_e).sum()
    if union == 0:
        return 1.0
    return float(inter) / float(union)


def compute_lpips(pred_gray: np.ndarray, gt_gray: np.ndarray, lpips_model, device: torch.device) -> float:
    if lpips_model is None:
        return float("nan")

    pred = torch.from_numpy(pred_gray).unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.float32)
    gt = torch.from_numpy(gt_gray).unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.float32)
    pred = pred.repeat(1, 3, 1, 1) * 2.0 - 1.0
    gt = gt.repeat(1, 3, 1, 1) * 2.0 - 1.0
    with torch.no_grad():
        val = lpips_model(pred, gt).item()
    if not np.isfinite(val):
        return float("nan")
    return float(val)


def save_progress_visual(
    out_path: Path,
    rgb: torch.Tensor,
    enhanced_rgb: torch.Tensor,
    z_t: torch.Tensor,
    pred_ir: torch.Tensor,
    gt_ir: torch.Tensor,
    metric_text: str,
) -> None:
    labels = ["Input RGB", "Enhanced", "Interpolated z_t", "Prediction", "Ground Truth"]
    imgs = [
        to_rgb_uint8(rgb),
        to_rgb_uint8(enhanced_rgb),
        normalize_gray_for_viz(z_t),
        to_rgb_uint8(pred_ir),
        to_rgb_uint8(gt_ir),
    ]

    h, w, _ = imgs[0].shape
    header_h = 28
    footer_h = 28
    canvas = Image.new("RGB", (w * 5, h + header_h + footer_h), color=(20, 20, 20))
    draw = ImageDraw.Draw(canvas)

    for i, (label, im) in enumerate(zip(labels, imgs)):
        x0 = i * w
        canvas.paste(Image.fromarray(im), (x0, header_h))
        draw.text((x0 + 8, 8), label, fill=(240, 240, 240))

    draw.text((8, h + header_h + 6), metric_text, fill=(240, 240, 240))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def save_epoch_progress_visuals(
    model: MVGSBModel,
    epoch: int,
    cfg: TrainConfig,
    device: torch.device,
    progress_samples: list[dict[str, torch.Tensor | str]],
    lpips_model,
) -> None:
    out_dir = cfg.project_root / cfg.progress_eval_outdir / f"epoch_{epoch:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    was_training = model.training
    model.eval()

    with torch.no_grad():
        for sample in progress_samples:
            sample_id = str(sample["id"])
            rgb = sample["rgb"].unsqueeze(0).to(device)
            ir = sample["ir"].unsqueeze(0).to(device)
            if device.type == "cuda":
                rgb = rgb.contiguous(memory_format=torch.channels_last)
                ir = ir.contiguous(memory_format=torch.channels_last)

            t = torch.full((1,), float(cfg.progress_eval_t), device=device)
            out = model(rgb=rgb, ir=ir, t=t, semantic_scale=0.0, task_scale=0.0)

            pred_gray = np.clip(out["pred_ir"][0].detach().cpu().numpy().squeeze(), 0.0, 1.0)
            gt_gray = np.clip(ir[0].detach().cpu().numpy().squeeze(), 0.0, 1.0)
            psnr = float(peak_signal_noise_ratio(gt_gray, pred_gray, data_range=1.0))
            ssim = float(structural_similarity(gt_gray, pred_gray, data_range=1.0))
            canny_iou = compute_canny_iou(pred_gray, gt_gray, sigma=1.0)
            lpips_val = compute_lpips(pred_gray, gt_gray, lpips_model, device=device)

            lpips_text = f"{lpips_val:.4f}" if np.isfinite(lpips_val) else "N/A"
            metric_text = f"PSNR={psnr:.3f}  SSIM={ssim:.4f}  Canny-IoU={canny_iou:.4f}  LPIPS={lpips_text}"

            save_progress_visual(
                out_path=out_dir / f"{sample_id}.png",
                rgb=rgb[0],
                enhanced_rgb=out["enhanced_rgb"][0],
                z_t=out["z_t"][0],
                pred_ir=out["pred_ir"][0],
                gt_ir=ir[0],
                metric_text=metric_text,
            )

    if was_training:
        model.train()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MV-GSB on LLVIP")
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        help="all|1|2|3|stage1|stage2|stage3|StructuralWarmup|SemanticAnchoring|TaskAwareRefinement",
    )
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--use-real-dino", action="store_true")
    parser.add_argument("--disable-cached-dino", action="store_true")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=None)
    parser.add_argument("--amp-dtype", type=str, default=None, help="float16|bfloat16")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--detect-anomaly", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainConfig()

    if args.dataset_root:
        cfg.dataset_root = Path(args.dataset_root)
    if args.output_root:
        cfg.output_root = Path(args.output_root)
    if args.max_epochs is not None:
        cfg.max_epochs = int(args.max_epochs)
    if args.batch_size is not None:
        cfg.batch_size = int(args.batch_size)
    if args.num_workers is not None:
        cfg.num_workers = int(args.num_workers)
    if args.grad_accum_steps is not None:
        cfg.grad_accum_steps = int(args.grad_accum_steps)
    if args.amp_dtype is not None:
        cfg.amp_dtype = args.amp_dtype
    if args.compile:
        cfg.use_torch_compile = True
    if args.detect_anomaly:
        cfg.detect_anomaly = True
    if args.disable_cached_dino:
        cfg.use_cached_dino_features = False
    if args.use_real_dino:
        cfg.use_real_dino = True

    set_seed(cfg.seed)

    torch.set_float32_matmul_precision(cfg.matmul_precision)
    torch.autograd.set_detect_anomaly(cfg.detect_anomaly)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model = MVGSBModel(
        sigma_max=cfg.sigma_max,
        use_real_dino=cfg.use_real_dino,
        dino_model_name=cfg.dino_model_name,
        time_eps=cfg.time_eps,
    ).to(device=device, memory_format=torch.channels_last)

    if cfg.use_torch_compile and hasattr(torch, "compile"):
        model = torch.compile(model, mode="reduce-overhead")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.use_amp and cfg.amp_dtype == "float16" and device.type == "cuda"))

    train_loader = build_loader(cfg, split="train")
    progress_samples = build_progress_samples(cfg)
    lpips_model = try_build_lpips_model(cfg.progress_eval_use_lpips, device=device)
    if cfg.progress_eval_use_lpips and lpips_model is None:
        print("[warn] LPIPS requested for progress visuals but unavailable; rendering LPIPS=N/A")

    output_dir = cfg.project_root / cfg.output_root
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "training_metrics_v2.csv"
    write_header = not metrics_path.exists()

    global_step = 0
    start_epoch = 1

    if args.resume is not None:
        ckpt = torch.load(Path(args.resume), map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"], strict=False)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
            for group in optimizer.param_groups:
                group["lr"] = cfg.lr
                group["weight_decay"] = cfg.weight_decay
        start_epoch = int(ckpt.get("epoch", 0)) + 1

    epoch_start, epoch_end = get_epoch_range_for_stage(cfg, args.stage)
    start_epoch = max(start_epoch, epoch_start)
    epoch_end = min(epoch_end, cfg.max_epochs)

    optimizer.zero_grad(set_to_none=True)

    for epoch in range(start_epoch, epoch_end + 1):
        model.train()
        stage = get_stage_for_epoch(cfg, epoch)

        sem_scale = stage.semantic_weight * curriculum_scale(epoch, warmup_epochs=50)
        task_scale = stage.task_weight * curriculum_scale(epoch, warmup_epochs=50)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d} | {stage.name}")

        running = {
            "total": 0.0,
            "noise": 0.0,
            "edge": 0.0,
            "semantic": 0.0,
            "task": 0.0,
            "entropy": 0.0,
            "enh_mean": 0.0,
            "enh_std": 0.0,
            "noise_std": 0.0,
        }

        for i, batch in enumerate(pbar, start=1):
            rgb = batch["rgb"].to(device, non_blocking=True)
            ir = batch["ir"].to(device, non_blocking=True)
            if device.type == "cuda":
                rgb = rgb.contiguous(memory_format=torch.channels_last)
                ir = ir.contiguous(memory_format=torch.channels_last)
            dino_feat_rgb = batch.get("dino_feat_rgb")
            if dino_feat_rgb is not None:
                dino_feat_rgb = dino_feat_rgb.to(device, non_blocking=True)
            t = torch.rand(rgb.size(0), device=device)

            use_autocast = cfg.use_amp and device.type == "cuda"
            amp_dtype = torch.bfloat16 if cfg.amp_dtype == "bfloat16" else torch.float16

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_autocast):
                outputs = model(
                    rgb=rgb,
                    ir=ir,
                    t=t,
                    semantic_scale=sem_scale if stage.use_semantic_guidance else 0.0,
                    task_scale=task_scale if stage.use_task_guidance else 0.0,
                    dino_feat_rgb=dino_feat_rgb,
                )
                losses = composite_loss(
                    outputs,
                    noise_weight=cfg.noise_loss_weight,
                    edge_weight=cfg.edge_loss_weight,
                    entropy_weight=cfg.entropy_kl_weight,
                )
                loss = losses["total"] / cfg.grad_accum_steps

            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                continue

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if i % cfg.grad_accum_steps == 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.max_grad_norm)
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            global_step += 1
            for k in running:
                if k in losses:
                    running[k] += losses[k].detach().item()

            running["enh_mean"] += outputs["enhanced_rgb"].detach().mean().item()
            running["enh_std"] += outputs["enhanced_rgb"].detach().std(unbiased=False).item()
            running["noise_std"] += outputs["pred_noise"].detach().std(unbiased=False).item()

            pbar.set_postfix(
                total=f"{running['total'] / i:.4f}",
                noise=f"{running['noise'] / i:.4f}",
                edge=f"{running['edge'] / i:.4f}",
                sem=f"{running['semantic'] / i:.4f}",
                task=f"{running['task'] / i:.4f}",
            )

        if len(train_loader) % cfg.grad_accum_steps != 0:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.max_grad_norm)
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg,
        }
        torch.save(ckpt, output_dir / f"mvgsb_epoch_{epoch:03d}.pt")

        num_batches = max(len(train_loader), 1)
        avg_total = running["total"] / num_batches
        avg_noise = running["noise"] / num_batches
        avg_edge = running["edge"] / num_batches
        avg_semantic = running["semantic"] / num_batches
        avg_task = running["task"] / num_batches
        avg_entropy = running["entropy"] / num_batches
        avg_enh_mean = running["enh_mean"] / num_batches
        avg_enh_std = running["enh_std"] / num_batches
        avg_noise_std = running["noise_std"] / num_batches

        with metrics_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(
                    [
                        "epoch",
                        "stage",
                        "total",
                        "noise",
                        "edge",
                        "semantic",
                        "task",
                        "entropy",
                        "enh_mean",
                        "enh_std",
                        "pred_noise_std",
                    ]
                )
                write_header = False
            writer.writerow(
                [
                    epoch,
                    stage.name,
                    f"{avg_total:.6f}",
                    f"{avg_noise:.6f}",
                    f"{avg_edge:.6f}",
                    f"{avg_semantic:.6f}",
                    f"{avg_task:.6f}",
                    f"{avg_entropy:.6f}",
                    f"{avg_enh_mean:.6f}",
                    f"{avg_enh_std:.6f}",
                    f"{avg_noise_std:.6f}",
                ]
            )

        if cfg.progress_eval_every > 0 and epoch % cfg.progress_eval_every == 0:
            save_epoch_progress_visuals(
                model=model,
                epoch=epoch,
                cfg=cfg,
                device=device,
                progress_samples=progress_samples,
                lpips_model=lpips_model,
            )


if __name__ == "__main__":
    main()
