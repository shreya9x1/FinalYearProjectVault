from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


class LLVIPPairedDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        image_size: int = 512,
        dino_cache_dir: str | Path | None = None,
    ):
        self.root = Path(root)
        self.split = split
        self.visible_dir = self.root / "visible" / split
        self.infrared_dir = self.root / "infrared" / split
        self.dino_cache_dir = Path(dino_cache_dir) if dino_cache_dir is not None else None

        if not self.visible_dir.exists() or not self.infrared_dir.exists():
            raise FileNotFoundError(
                f"Expected split folders at {self.visible_dir} and {self.infrared_dir}"
            )

        self.transform = T.Compose(
            [
                T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
                T.ToTensor(),
            ]
        )

        self.pairs = self._build_pairs()
        if not self.pairs:
            raise RuntimeError(f"No LLVIP pairs found in split={split} under {self.root}")

    def _build_pairs(self) -> List[Dict[str, Path]]:
        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        visible_paths: List[Path] = []
        infra_paths: List[Path] = []

        for ext in exts:
            visible_paths.extend(self.visible_dir.glob(ext))
            infra_paths.extend(self.infrared_dir.glob(ext))

        visible_paths = sorted(visible_paths)
        infra_index = {p.stem: p for p in infra_paths}

        pairs: List[Dict[str, Path]] = []
        for vp in visible_paths:
            ip = infra_index.get(vp.stem)
            if ip is None:
                continue
            pairs.append({"id": vp.stem, "visible": vp, "infrared": ip})
        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        rec = self.pairs[index]
        rgb = Image.open(rec["visible"]).convert("RGB")
        ir = Image.open(rec["infrared"]).convert("L")

        rgb_tensor = self.transform(rgb)
        ir_tensor = self.transform(ir)

        sample: Dict[str, torch.Tensor | str] = {
            "id": rec["id"],
            "rgb": rgb_tensor,
            "ir": ir_tensor,
        }

        if self.dino_cache_dir is not None:
            feat_path = self.dino_cache_dir / f"{rec['id']}.pt"
            if feat_path.exists():
                sample["dino_feat_rgb"] = torch.load(feat_path, map_location="cpu")

        return sample
