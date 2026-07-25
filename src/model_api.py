"""Model integration layer for DP Naturals.

Loads the DINOv2-L LoRA multi-task checkpoint trained in dp_final.ipynb and
exposes a single `predict(image)` entry point that returns the top-k dish
predictions, nutrition lookup, and allergen badge codes for the UI.

Paths default to the assignment3 project root (sibling to this dp_naturals
folder); override via env vars if needed:
    DP_MODEL_CKPT       path to the .pt checkpoint
    DP_NUTRITION_DB     path to nutrition_db.json
    DP_DEVICE           'cuda', 'cuda:0', 'cpu', ...      (default: auto)
    DP_T_VAL            temperature for softmax           (default: 1.0)
    DP_OOD_THRESHOLD    MSP threshold for OOD flag        (default: 0.30)
"""
from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


# Project root = parent of this dp_naturals/ folder.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CKPT = str(_PROJECT_ROOT / "models" / "dinov2_large_best.pt")
DEFAULT_NUTRITION = str(_PROJECT_ROOT / "data" / "nutrition" / "nutrition_db.json")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class AttentivePool(nn.Module):
    def __init__(self, dim: int, heads: int = 8):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, num_heads=heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        q = self.q.expand(tokens.size(0), -1, -1)
        out, _ = self.attn(q, tokens, tokens)
        return self.norm(out.squeeze(1))


class MultiTaskHead(nn.Module):
    def __init__(self, dim: int, n_dish: int, n_diet: int = 3, n_course: int = 4, dropout: float = 0.2):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.dish = nn.Linear(dim, n_dish)
        self.diet = nn.Linear(dim, n_diet)
        self.course = nn.Linear(dim, n_course)

    def forward(self, feat: torch.Tensor):
        feat = self.drop(feat)
        return self.dish(feat), self.diet(feat), self.course(feat)


class FoodModel(nn.Module):
    """Same arch as dp_final.ipynb cell 7, restricted to the DINOv2 backbones used at inference."""

    def __init__(self, backbone_name: str, n_dish: int, lora_r: int = 16):
        super().__init__()
        self.backbone_name = backbone_name
        if backbone_name in ("dinov2_large", "dinov2_base"):
            from transformers import AutoConfig, AutoModel
            from peft import LoraConfig, get_peft_model

            hf_id = {"dinov2_large": "facebook/dinov2-large", "dinov2_base": "facebook/dinov2-base"}[backbone_name]
            # Build architecture from config only (no 1.2 GB weight download).
            # The checkpoint we load later contains all backbone + LoRA weights.
            config = AutoConfig.from_pretrained(hf_id)
            base = AutoModel.from_config(config)
            for p in base.parameters():
                p.requires_grad = False
            cfg = LoraConfig(r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.05,
                             target_modules=["query", "value"], bias="none")
            self.backbone = get_peft_model(base, cfg)
            dim = 1024 if backbone_name == "dinov2_large" else 768
            self.pool = AttentivePool(dim)
        else:
            import timm
            timm_id = {
                "efficientnet_b3": "efficientnet_b3",
                "convnextv2_base": "convnextv2_base.fcmae_ft_in22k_in1k_384",
                "convnextv2_small": "convnextv2_small.fcmae_ft_in22k_in1k",
            }[backbone_name]
            self.backbone = timm.create_model(timm_id, pretrained=True, num_classes=0, global_pool="avg")
            dim = self.backbone.num_features
            self.pool = nn.Identity()
        self.head = MultiTaskHead(dim, n_dish)

    def forward(self, x: torch.Tensor):
        if self.backbone_name.startswith("dinov2"):
            out = self.backbone(pixel_values=x)
            tokens = out.last_hidden_state[:, 1:, :]
            feat = self.pool(tokens)
        else:
            feat = self.backbone(x)
        return self.head(feat)


def _build_eval_transform(img_size: int):
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _pretty_class(name: str) -> str:
    return name.replace("_", " ").title()


class DishRecognizer:
    """Wraps the trained model + nutrition DB. Construct once, call predict() per request."""

    def __init__(
        self,
        ckpt_path: str | None = None,
        nutrition_path: str | None = None,
        device: str | None = None,
        t_val: float | None = None,
        ood_threshold: float | None = None,
    ):
        self.ckpt_path = Path(ckpt_path or os.environ.get("DP_MODEL_CKPT", DEFAULT_CKPT))
        self.nutrition_path = Path(nutrition_path or os.environ.get("DP_NUTRITION_DB", DEFAULT_NUTRITION))
        self.device = device or os.environ.get("DP_DEVICE") or self._pick_device()
        self.t_val = float(t_val if t_val is not None else os.environ.get("DP_T_VAL", 1.0))
        self.ood_threshold = float(ood_threshold if ood_threshold is not None else os.environ.get("DP_OOD_THRESHOLD", 0.30))

        if not self.ckpt_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found at {self.ckpt_path}. "
                "Set DP_MODEL_CKPT to the trained .pt file from dp_final.ipynb."
            )
        if not self.nutrition_path.exists():
            raise FileNotFoundError(
                f"Nutrition DB not found at {self.nutrition_path}. "
                "Set DP_NUTRITION_DB to nutrition_db.json from dp_final.ipynb."
            )

        ckpt = torch.load(self.ckpt_path, map_location=self.device)
        self.classes: list[str] = ckpt["classes"]
        self.backbone: str = ckpt.get("backbone", "dinov2_large")
        self.img_size: int = int(ckpt.get("img_size", 518))
        self.val_top1: float = float(ckpt.get("val_top1", 0.0))

        self.model = FoodModel(self.backbone, n_dish=len(self.classes)).to(self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

        self.transform = _build_eval_transform(self.img_size)
        self.nutrition_db: dict[str, dict[str, Any]] = json.loads(self.nutrition_path.read_text())

    @staticmethod
    def _pick_device(min_free_gb: float = 3.0) -> str:
        """Pick a CUDA device that has at least `min_free_gb` free, else CPU.

        DINOv2-L at 518² needs ~3 GB just for inference, so a fragmented or
        shared GPU (e.g. one that's also hosting a Jupyter kernel) would OOM.
        """
        if not torch.cuda.is_available():
            return "cpu"
        for i in range(torch.cuda.device_count()):
            try:
                free, _ = torch.cuda.mem_get_info(i)
                if free / 1e9 >= min_free_gb:
                    return f"cuda:{i}"
            except Exception:
                continue
        return "cpu"

    @torch.no_grad()
    def predict(self, image: Image.Image | bytes | str | Path, k: int = 5) -> dict[str, Any]:
        if isinstance(image, (bytes, bytearray)):
            img = Image.open(io.BytesIO(image)).convert("RGB")
        elif isinstance(image, (str, Path)):
            img = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            img = image.convert("RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(image)!r}")

        x = self.transform(img).unsqueeze(0).to(self.device)
        use_amp = self.device.startswith("cuda")
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, _, _ = self.model(x)
        else:
            logits, _, _ = self.model(x)
        probs = F.softmax(logits.float() / self.t_val, dim=1)[0].cpu().numpy()

        top_idx = probs.argsort()[::-1][:k]
        top = [
            {
                "class": self.classes[int(i)],
                "label": _pretty_class(self.classes[int(i)]),
                "confidence": float(probs[int(i)]),
            }
            for i in top_idx
        ]
        msp = float(probs.max())
        is_ood = msp < self.ood_threshold

        result: dict[str, Any] = {
            "predictions": top,
            "top": top[0],
            "msp": msp,
            "is_ood": is_ood,
            "model_val_top1": self.val_top1,
        }

        if is_ood:
            result["message"] = "Low confidence — image may not be Indian food."
            result["nutrition"] = None
            result["allergens"] = None
            return result

        top_class = top[0]["class"]
        nutri = self.nutrition_db.get(top_class, {}) or {}
        result["nutrition"] = self._format_nutrition(nutri)
        result["allergens"] = self._format_allergens(nutri.get("allergens") or {})
        result["meta"] = {
            "diet": nutri.get("diet", "unknown"),
            "course": nutri.get("course", "unknown"),
            "region": nutri.get("region", "India"),
            "provenance": nutri.get("provenance", {}),
        }
        return result

    @staticmethod
    def _format_nutrition(entry: dict[str, Any]) -> dict[str, Any]:
        keys = [("kcal", "Calories", "kcal"), ("carb", "Carbs", "g"),
                ("protein", "Protein", "g"), ("fat", "Fat", "g")]
        out: dict[str, Any] = {}
        for key, label, unit in keys:
            v = entry.get(key)
            if not isinstance(v, dict):
                continue
            out[key] = {
                "label": label,
                "unit": unit,
                "point": round(float(v.get("point", 0)), 1),
                "low": round(float(v.get("low", 0)), 1),
                "high": round(float(v.get("high", 0)), 1),
            }
        return out

    @staticmethod
    def _format_allergens(entry: dict[str, Any]) -> list[dict[str, str]]:
        order = [("peanuts", "Peanuts"), ("dairy", "Dairy"), ("gluten", "Gluten")]
        legend = {
            "red": "Contains allergen (high risk)",
            "yellow": "May contain / uncertain",
            "green": "Safe / not present",
        }
        out: list[dict[str, str]] = []
        for key, label in order:
            code = entry.get(key)
            if code not in ("red", "yellow", "green"):
                code = "unknown"
            out.append({
                "key": key,
                "label": label,
                "code": code,
                "description": legend.get(code, "Unknown"),
            })
        return out


_singleton: DishRecognizer | None = None


def get_recognizer() -> DishRecognizer:
    """Lazily build a process-wide recognizer so the model is loaded once."""
    global _singleton
    if _singleton is None:
        _singleton = DishRecognizer()
    return _singleton
