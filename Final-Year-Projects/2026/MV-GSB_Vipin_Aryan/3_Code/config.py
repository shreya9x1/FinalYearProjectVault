from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict


@dataclass
class StageConfig:
    name: str
    start_epoch: int
    end_epoch: int
    use_semantic_guidance: bool
    use_task_guidance: bool
    semantic_weight: float
    task_weight: float


@dataclass
class TrainConfig:
    project_root: Path = Path(__file__).resolve().parent
    dataset_root: Path = Path(r"C:\Project\FYP2.0\LLVIP")
    output_root: Path = Path("outputs")

    image_size: int = 512
    batch_size: int = 2
    grad_accum_steps: int = 8
    num_workers: int = 8
    prefetch_factor: int = 4
    persistent_workers: bool = True

    max_epochs: int = 300
    lr: float = 1e-5
    weight_decay: float = 1e-4
    seed: int = 42

    # Hardware-focused settings for RTX 2000 Ada style constraints.
    use_amp: bool = True
    amp_dtype: str = "float16"
    matmul_precision: str = "high"
    gradient_checkpointing: bool = True
    use_torch_compile: bool = False
    detect_anomaly: bool = False

    # Bridge schedule
    max_timesteps: int = 1000
    sigma_max: float = 0.25
    time_eps: float = 1e-4

    # Optimization stability
    max_grad_norm: float = 1.0

    # Loss weights
    noise_loss_weight: float = 1.0
    edge_loss_weight: float = 1.0
    entropy_kl_weight: float = 1e-3

    # Oracle feature cache
    dino_cache_dir: Path = Path("cache") / "dinov2"
    use_cached_dino_features: bool = True
    use_real_dino: bool = False
    dino_model_name: str = "dinov2_vitl14"

    # Progress visualization/evaluation callback
    progress_eval_every: int = 5
    progress_eval_num_samples: int = 2
    progress_eval_split: str = "test"
    progress_eval_t: float = 0.5
    progress_eval_outdir: Path = Path("outputs") / "progress_viz"
    progress_eval_use_lpips: bool = True

    stages: Dict[str, StageConfig] = field(
        default_factory=lambda: {
            "stage1": StageConfig(
                name="StructuralWarmup",
                start_epoch=1,
                end_epoch=41,
                use_semantic_guidance=False,
                use_task_guidance=False,
                semantic_weight=0.0,
                task_weight=0.0,
            ),
            "stage2": StageConfig(
                name="SemanticAnchoring",
                start_epoch=42,
                end_epoch=150,
                use_semantic_guidance=True,
                use_task_guidance=False,
                semantic_weight=0.2,
                task_weight=0.0,
            ),
            "stage3": StageConfig(
                name="TaskAwareRefinement",
                start_epoch=151,
                end_epoch=300,
                use_semantic_guidance=True,
                use_task_guidance=True,
                semantic_weight=0.2,
                task_weight=0.3,
            ),
        }
    )


def get_stage_for_epoch(cfg: TrainConfig, epoch: int) -> StageConfig:
    for stage in cfg.stages.values():
        if stage.start_epoch <= epoch <= stage.end_epoch:
            return stage
    return cfg.stages["stage3"]


def get_epoch_range_for_stage(cfg: TrainConfig, stage_name: str) -> tuple[int, int]:
    key = stage_name.lower().strip().replace("_", "").replace("-", "").replace(" ", "")
    if key == "all":
        return 1, cfg.max_epochs
    if key in {"1", "stage1"}:
        stage = cfg.stages["stage1"]
        return stage.start_epoch, stage.end_epoch
    if key in {"2", "stage2"}:
        stage = cfg.stages["stage2"]
        return stage.start_epoch, stage.end_epoch
    if key in {"3", "stage3"}:
        stage = cfg.stages["stage3"]
        return stage.start_epoch, stage.end_epoch
    if key == "structuralwarmup":
        stage = cfg.stages["stage1"]
        return stage.start_epoch, stage.end_epoch
    if key == "semanticanchoring":
        stage = cfg.stages["stage2"]
        return stage.start_epoch, stage.end_epoch
    if key == "taskawarerefinement":
        stage = cfg.stages["stage3"]
        return stage.start_epoch, stage.end_epoch
    raise ValueError(f"Unsupported stage selector: {stage_name}")
