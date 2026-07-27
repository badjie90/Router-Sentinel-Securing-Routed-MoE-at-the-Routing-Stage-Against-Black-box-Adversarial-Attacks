#!/usr/bin/env python3
"""Adversarial robustification for the BDD100K MoE router/gating network."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import random
import sys
import time
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

NUM_ROUTER_CLASSES = 3
EXPERT_INDEX_TO_NAME = {0: "weather", 1: "scene", 2: "time"}
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

DEFAULT_TRAIN_SCRIPT = "/mnt/nvme1n1/bbadjie/SEAS/AAutonomous/Baseline/BDD100k/scripts/bdd100k_moe_train.py"
DEFAULT_METADATA_DIR = "/mnt/nvme1n1/bbadjie/SEAS/AAutonomous/Baseline/BDD100k/data/metadata_files/metadata-New"
DEFAULT_RUN_GLOB = "/mnt/nvme1n1/bbadjie/SEAS/AAutonomous/Baseline/BDD100k/New-train-models/moe_stage3"
DEFAULT_OUTPUT_DIR = "/mnt/nvme1n1/bbadjie/SEAS/AAutonomous/Adversarial-Robustification/BDD100k/router_sentinel"


def timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def import_module_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import module from: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def ensure_train_import_dependencies() -> None:
    """Allow architecture-only imports when torchmetrics is unavailable."""
    try:
        import torchmetrics.classification  # noqa: F401
        return
    except ImportError:
        pass

    class _UnavailableMetric:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("torchmetrics is unavailable in this environment.")

    torchmetrics_mod = types.ModuleType("torchmetrics")
    classification_mod = types.ModuleType("torchmetrics.classification")
    classification_mod.MultilabelAUROC = _UnavailableMetric
    classification_mod.MultilabelF1Score = _UnavailableMetric
    classification_mod.MulticlassAccuracy = _UnavailableMetric
    torchmetrics_mod.classification = classification_mod
    sys.modules.setdefault("torchmetrics", torchmetrics_mod)
    sys.modules.setdefault("torchmetrics.classification", classification_mod)


def list_run_dirs(run_dirs: Sequence[str], run_glob: str) -> List[Path]:
    if run_dirs:
        resolved = [Path(p).expanduser().resolve() for p in run_dirs]
    else:
        glob_path = Path(run_glob).expanduser()
        resolved = sorted(glob_path.parent.glob(glob_path.name)) if glob_path.is_absolute() else sorted(Path(".").glob(run_glob))
        resolved = [p.resolve() for p in resolved]
    resolved = [p for p in resolved if p.exists()]
    if not resolved:
        raise FileNotFoundError(f"No run directories found. Got --run-glob={run_glob!r}")
    return resolved


class BDD100KPixelDataset(Dataset):
    """BDD100K rows as unnormalized pixel-space tensors in [0, 1]."""

    def __init__(self, rows: Sequence[Dict[str, Any]], image_size: int) -> None:
        self.rows = list(rows)
        self.transform = transforms.Compose(
            [
                transforms.Resize(int(image_size * 1.14)),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.rows[index]
        image = Image.open(row["image_path"]).convert("RGB")
        return {
            "image": self.transform(image),
            "objects": torch.tensor([row["car_present"], row["pedestrian_present"], row["traffic_sign_present"]], dtype=torch.float32),
            "image_path": row["image_path"],
        }


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int, device: torch.device) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )


def load_base_model(
    run_dir: Path,
    checkpoint_name: str,
    train_module,
    metadata_bundle: Dict[str, Any],
    device: torch.device,
) -> Tuple[nn.Module, Dict[str, Any]]:
    config = load_json(run_dir / "config.json")
    checkpoint_path = run_dir / "checkpoints" / checkpoint_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    weather_names = [k for k, _ in sorted(metadata_bundle["weather_to_id"].items(), key=lambda kv: kv[1])]
    scene_names = [k for k, _ in sorted(metadata_bundle["scene_to_id"].items(), key=lambda kv: kv[1])]
    time_names = [k for k, _ in sorted(metadata_bundle["time_to_id"].items(), key=lambda kv: kv[1])]
    stage = str(config.get("stage", "stage3_moe"))
    if stage == "stage0_baseline":
        raise ValueError("Router robustification requires a MoE checkpoint with a router.")
    model = train_module.build_model(
        stage=stage,
        backbone_name=str(config.get("backbone_name", "convnextv2_tiny.fcmae_ft_in1k")),
        num_weather=len(weather_names),
        num_scene=len(scene_names),
        num_time=len(time_names),
        pretrained=False,
    )
    payload = torch.load(checkpoint_path, map_location="cpu")
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.load_state_dict(state, strict=True)
    model.to(device)
    return model, config


def routing_entropy(probs: torch.Tensor) -> torch.Tensor:
    probs = probs.clamp_min(1e-8)
    return -torch.sum(probs * torch.log(probs), dim=1)


def top1_top2_margin(logits: torch.Tensor) -> torch.Tensor:
    top2 = torch.topk(logits, k=2, dim=1).values
    return top2[:, 0] - top2[:, 1]


def energy_score(logits: torch.Tensor) -> torch.Tensor:
    return -torch.logsumexp(logits, dim=1)


def js_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    p = p.clamp_min(1e-8)
    q = q.clamp_min(1e-8)
    m = 0.5 * (p + q)
    return 0.5 * torch.sum(p * (torch.log(p) - torch.log(m)), dim=1) + 0.5 * torch.sum(q * (torch.log(q) - torch.log(m)), dim=1)


def tiny_blur(x: torch.Tensor) -> torch.Tensor:
    kernel = torch.tensor([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]], device=x.device, dtype=x.dtype)
    kernel = (kernel / kernel.sum()).view(1, 1, 3, 3).repeat(x.size(1), 1, 1, 1)
    return F.conv2d(x, kernel, padding=1, groups=x.size(1))


def micro_transforms(x: torch.Tensor) -> List[torch.Tensor]:
    return [
        (x + 0.01 * torch.randn_like(x)).clamp(0.0, 1.0),
        torch.roll(x, shifts=1, dims=3),
        tiny_blur(x).clamp(0.0, 1.0),
        (x + 0.02).clamp(0.0, 1.0),
    ]


class RobustifiedBDD100KMoE(nn.Module):
    """Pixel-space wrapper that adds attack/risk heads to the trained MoE."""

    def __init__(self, base_model: nn.Module) -> None:
        super().__init__()
        self.base_model = base_model
        dim = int(base_model.backbone.out_dim)
        self.attack_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 128), nn.GELU(), nn.Dropout(0.2), nn.Linear(128, 2))
        self.risk_head = nn.Sequential(nn.LayerNorm(8), nn.Linear(8, 32), nn.GELU(), nn.Linear(32, 1))
        self.attack_temperature = nn.Parameter(torch.ones(1), requires_grad=False)
        self.register_buffer("mean", torch.tensor(IMAGENET_DEFAULT_MEAN, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_DEFAULT_STD, dtype=torch.float32).view(1, 3, 1, 1))

    def normalize(self, image: torch.Tensor) -> torch.Tensor:
        return (image - self.mean) / self.std

    def routing_only(self, image: torch.Tensor) -> torch.Tensor:
        return self.forward(image, audit=False)["router_logits"]

    def forward(self, image: torch.Tensor, audit: bool = True, calibrated_attack: bool = True) -> Dict[str, torch.Tensor]:
        outputs = self.base_model(self.normalize(image))
        features = outputs["shared_features"]
        attack_logits = self.attack_head(features)
        if calibrated_attack:
            attack_logits = attack_logits / self.attack_temperature.clamp_min(0.05)
        router_logits = outputs["router_logits"]
        alpha = torch.softmax(router_logits, dim=1)
        base = torch.stack(
            [
                torch.softmax(attack_logits, dim=1)[:, 1],
                routing_entropy(alpha),
                -top1_top2_margin(router_logits),
                -energy_score(router_logits),
                1.0 - alpha.max(dim=1).values,
            ],
            dim=1,
        )
        if not audit:
            audit_features = torch.zeros((image.size(0), 3), device=image.device, dtype=image.dtype)
        else:
            base_top1 = torch.argmax(alpha, dim=1)
            base_margin = top1_top2_margin(router_logits)
            rc_terms: List[torch.Tensor] = []
            jsd_terms: List[torch.Tensor] = []
            md_terms: List[torch.Tensor] = []
            for xm in micro_transforms(image):
                out_m = self.base_model(self.normalize(xm))
                logits_m = out_m["router_logits"]
                alpha_m = torch.softmax(logits_m, dim=1)
                rc_terms.append(torch.argmax(alpha_m, dim=1).eq(base_top1).float())
                jsd_terms.append(js_divergence(alpha, alpha_m))
                md_terms.append(base_margin - top1_top2_margin(logits_m))
            audit_features = torch.stack(
                [
                    1.0 - torch.stack(rc_terms).mean(dim=0),
                    torch.stack(jsd_terms).mean(dim=0),
                    torch.stack(md_terms).mean(dim=0),
                ],
                dim=1,
            )
        outputs["attack_logits"] = attack_logits
        outputs["risk"] = self.risk_head(torch.cat([base, audit_features], dim=1)).squeeze(1)
        return outputs


def freeze_for_router_robustification(model: RobustifiedBDD100KMoE, train_backbone: bool) -> None:
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.base_model.router.parameters():
        p.requires_grad_(True)
    for p in model.base_model.fused_obj_head.parameters():
        p.requires_grad_(True)
    for p in model.attack_head.parameters():
        p.requires_grad_(True)
    for p in model.risk_head.parameters():
        p.requires_grad_(True)
    if train_backbone:
        for p in model.base_model.backbone.parameters():
            p.requires_grad_(True)


def clamp_01(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, 0.0, 1.0)


def project_linf(x_adv: torch.Tensor, x_orig: torch.Tensor, eps: float) -> torch.Tensor:
    return clamp_01(torch.max(torch.min(x_adv, x_orig + eps), x_orig - eps))


def pgd_linf_router(model: RobustifiedBDD100KMoE, x: torch.Tensor, y_ref: torch.Tensor, eps: float, steps: int, step_size: float, random_start: bool) -> torch.Tensor:
    was_training = model.training
    model.eval()
    x_orig = x.detach()
    x_adv = project_linf(x_orig + torch.empty_like(x_orig).uniform_(-eps, eps), x_orig, eps) if random_start else x_orig.clone()
    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)
        loss = F.cross_entropy(model.routing_only(x_adv), y_ref)
        grad = torch.autograd.grad(loss, x_adv, only_inputs=True)[0]
        x_adv = project_linf(x_adv + step_size * torch.sign(grad), x_orig, eps)
    if was_training:
        model.train()
    return x_adv.detach()


@torch.no_grad()
def collect_clean_router_labels(model: RobustifiedBDD100KMoE, loader: DataLoader, device: torch.device) -> torch.Tensor:
    labels: List[torch.Tensor] = []
    model.eval()
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels.append(torch.argmax(model.routing_only(images), dim=1).cpu())
    return torch.cat(labels, dim=0).long()


def build_adv_loader(model: RobustifiedBDD100KMoE, loader: DataLoader, labels: torch.Tensor, device: torch.device, eps: float, steps: int, step_size: float) -> DataLoader:
    images_out: List[torch.Tensor] = []
    objects_out: List[torch.Tensor] = []
    offset = 0
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        n = int(x.size(0))
        y = labels[offset : offset + n].to(device, non_blocking=True)
        offset += n
        images_out.append(pgd_linf_router(model, x, y, eps, steps, step_size, random_start=True).cpu())
        objects_out.append(batch["objects"].cpu())

    class _TensorDictDataset(Dataset):
        def __init__(self, images: torch.Tensor, objects: torch.Tensor) -> None:
            self.images = images
            self.objects = objects

        def __len__(self) -> int:
            return int(self.images.size(0))

        def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
            return {"image": self.images[index], "objects": self.objects[index]}

    dataset = _TensorDictDataset(torch.cat(images_out, dim=0), torch.cat(objects_out, dim=0))
    return DataLoader(dataset, batch_size=loader.batch_size, shuffle=False, num_workers=0)


@torch.no_grad()
def evaluate(model: RobustifiedBDD100KMoE, loader: DataLoader, labels: torch.Tensor, device: torch.device, thresholds: Dict[str, float]) -> Dict[str, float]:
    model.eval()
    total = 0
    route_correct = 0
    allow = 0
    review = 0
    reject = 0
    offset = 0
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        n = int(x.size(0))
        y = labels[offset : offset + n].to(device, non_blocking=True)
        offset += n
        out = model(x, audit=True, calibrated_attack=True)
        pred = torch.argmax(out["router_logits"], dim=1)
        risk = out["risk"]
        allow_mask = risk < thresholds["allow"]
        reject_mask = risk >= thresholds["reject"]
        review_mask = (~allow_mask) & (~reject_mask)
        total += n
        route_correct += int(pred.eq(y).sum().item())
        allow += int(allow_mask.sum().item())
        review += int(review_mask.sum().item())
        reject += int(reject_mask.sum().item())
    return {
        "router_acc_vs_clean_top1": route_correct / max(1, total),
        "allow_rate": allow / max(1, total),
        "review_rate": review / max(1, total),
        "reject_rate": reject / max(1, total),
    }


@torch.no_grad()
def collect_risk_scores(model: RobustifiedBDD100KMoE, loader: DataLoader, device: torch.device) -> torch.Tensor:
    scores: List[torch.Tensor] = []
    model.eval()
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        scores.append(model(x, audit=True, calibrated_attack=True)["risk"].detach().cpu())
    return torch.cat(scores, dim=0)


def optimize_thresholds(clean_scores: torch.Tensor, adv_scores: torch.Tensor, clean_accept_target: float) -> Dict[str, float]:
    allow = float(torch.quantile(clean_scores, clean_accept_target).item())
    review = float(torch.quantile(clean_scores, min(0.98, clean_accept_target + 0.10)).item())
    reject = float(torch.quantile(adv_scores, 0.75).item())
    if reject <= review:
        reject = float(max(review + 1e-4, torch.quantile(adv_scores, 0.90).item()))
    return {"allow": allow, "review": review, "reject": reject}


@dataclass
class TrainConfig:
    train_script: str = DEFAULT_TRAIN_SCRIPT
    metadata_dir: str = DEFAULT_METADATA_DIR
    run_dirs: Tuple[str, ...] = ()
    run_glob: str = DEFAULT_RUN_GLOB
    checkpoint_name: str = "best.pt"
    output_dir: str = DEFAULT_OUTPUT_DIR
    max_train_samples: int = 4000
    max_val_samples: int = 1000
    val_ratio: float = 0.15
    batch_size: int = 16
    num_workers: int = 4
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    warmup_epochs: int = 2
    joint_epochs: int = 8
    lr: float = 1e-4
    sentinel_lr_mult: float = 5.0
    weight_decay: float = 1e-4
    attack_eps: float = 0.05
    attack_steps: int = 5
    attack_step_size: float = 0.0
    adv_fraction: float = 0.50
    clean_accept_target: float = 0.90
    train_backbone: bool = False


def train_one_run(cfg: TrainConfig, run_dir: Path, train_module, bundle: Dict[str, Any], rows: List[Dict[str, Any]], device: torch.device) -> None:
    run_out = Path(cfg.output_dir).resolve() / run_dir.name
    ensure_dir(run_out / "checkpoints")
    model_base, base_config = load_base_model(run_dir, cfg.checkpoint_name, train_module, bundle, device)
    model = RobustifiedBDD100KMoE(model_base).to(device)
    freeze_for_router_robustification(model, train_backbone=cfg.train_backbone)

    rng = random.Random(cfg.seed)
    rows = list(rows)
    rng.shuffle(rows)
    if cfg.max_train_samples > 0:
        rows = rows[: cfg.max_train_samples]
    n_val = min(max(1, int(round(len(rows) * cfg.val_ratio))), cfg.max_val_samples if cfg.max_val_samples > 0 else len(rows))
    val_rows = rows[:n_val]
    train_rows = rows[n_val:]
    image_size = int(base_config.get("image_size", 224))

    train_loader = make_loader(BDD100KPixelDataset(train_rows, image_size), cfg.batch_size, True, cfg.num_workers, device)
    val_loader = make_loader(BDD100KPixelDataset(val_rows, image_size), cfg.batch_size, False, cfg.num_workers, device)
    val_labels = collect_clean_router_labels(model, val_loader, device)

    step_size = cfg.attack_step_size if cfg.attack_step_size > 0 else (cfg.attack_eps / max(1, cfg.attack_steps)) * 2.5
    param_groups = [
        {"params": model.base_model.router.parameters(), "lr": cfg.lr},
        {"params": model.base_model.fused_obj_head.parameters(), "lr": cfg.lr},
        {"params": list(model.attack_head.parameters()) + list(model.risk_head.parameters()), "lr": cfg.lr * cfg.sentinel_lr_mult},
    ]
    if cfg.train_backbone:
        param_groups.append({"params": model.base_model.backbone.parameters(), "lr": cfg.lr * 0.1})
    optimizer = optim.AdamW(param_groups, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, cfg.warmup_epochs + cfg.joint_epochs), eta_min=1e-6)
    det_ce = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0], device=device))
    obj_bce = nn.BCEWithLogitsLoss()
    risk_bce = nn.BCEWithLogitsLoss()

    best_score = -1e9
    thresholds = {"allow": 1e9, "review": 1e9, "reject": 1e9}
    history: List[Dict[str, Any]] = []
    total_epochs = cfg.warmup_epochs + cfg.joint_epochs

    for epoch in range(1, total_epochs + 1):
        warmup = epoch <= cfg.warmup_epochs
        model.train()
        losses: List[float] = []
        route_accs: List[float] = []
        for batch_idx, batch in enumerate(train_loader, start=1):
            x = batch["image"].to(device, non_blocking=True)
            objects = batch["objects"].to(device, non_blocking=True)
            with torch.no_grad():
                y_ref = torch.argmax(model.routing_only(x), dim=1)
            n_adv = max(1, int(round(x.size(0) * cfg.adv_fraction)))
            adv_idx = torch.randperm(x.size(0), device=device)[:n_adv]
            x_adv = pgd_linf_router(model, x[adv_idx], y_ref[adv_idx], cfg.attack_eps, cfg.attack_steps, step_size, random_start=True)

            clean_out = model(x, audit=True, calibrated_attack=False)
            adv_out = model(x_adv, audit=True, calibrated_attack=False)
            route_loss = F.cross_entropy(clean_out["router_logits"], y_ref)
            adv_route_loss = F.cross_entropy(adv_out["router_logits"], y_ref[adv_idx])
            det_loss = det_ce(clean_out["attack_logits"], torch.zeros(x.size(0), dtype=torch.long, device=device))
            det_loss = det_loss + det_ce(adv_out["attack_logits"], torch.ones(x_adv.size(0), dtype=torch.long, device=device))
            risk_loss = risk_bce(clean_out["risk"], torch.zeros(x.size(0), device=device))
            risk_loss = risk_loss + risk_bce(adv_out["risk"], torch.ones(x_adv.size(0), device=device))
            clean_alpha = torch.softmax(clean_out["router_logits"].detach(), dim=1)
            adv_log_alpha = torch.log_softmax(adv_out["router_logits"], dim=1)
            consistency_loss = F.kl_div(adv_log_alpha, clean_alpha[adv_idx], reduction="batchmean")
            task_loss = obj_bce(clean_out["obj_fused_logits"], objects)
            loss = 0.50 * det_loss + 0.25 * risk_loss + 0.50 * task_loss
            if not warmup:
                loss = loss + 1.00 * route_loss + 1.25 * adv_route_loss + 0.25 * consistency_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.item()))
            route_accs.append(float(torch.argmax(clean_out["router_logits"], dim=1).eq(y_ref).float().mean().item()))
            if batch_idx % 25 == 0:
                print(f"[{timestamp()}] {run_dir.name} epoch {epoch}/{total_epochs} batch {batch_idx}/{len(train_loader)} loss={np.mean(losses):.5f}", flush=True)

        scheduler.step()
        adv_val_loader = build_adv_loader(model, val_loader, val_labels, device, cfg.attack_eps, cfg.attack_steps, step_size)
        thresholds = optimize_thresholds(collect_risk_scores(model, val_loader, device), collect_risk_scores(model, adv_val_loader, device), cfg.clean_accept_target)
        clean_metrics = evaluate(model, val_loader, val_labels, device, thresholds)
        adv_metrics = evaluate(model, adv_val_loader, val_labels, device, thresholds)
        score = clean_metrics["allow_rate"] - clean_metrics["reject_rate"] + adv_metrics["review_rate"] + adv_metrics["reject_rate"]
        row = {
            "epoch": epoch,
            "stage": "warmup" if warmup else "joint",
            "train_loss": float(np.mean(losses)),
            "train_route_acc_vs_pseudo": float(np.mean(route_accs)),
            "val_score": float(score),
            **{f"clean_{k}": v for k, v in clean_metrics.items()},
            **{f"adv_{k}": v for k, v in adv_metrics.items()},
            "tau_allow": thresholds["allow"],
            "tau_review": thresholds["review"],
            "tau_reject": thresholds["reject"],
        }
        history.append(row)
        write_csv(run_out / "training_history.csv", history)
        print(json.dumps(row), flush=True)

        payload = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "base_config": base_config,
            "thresholds": thresholds,
            "config": asdict(cfg),
            "run_dir": str(run_dir),
        }
        torch.save(payload, run_out / "checkpoints" / "last.pt")
        if score > best_score:
            best_score = score
            torch.save(payload, run_out / "checkpoints" / "best.pt")

    save_json({"best_score": best_score, "thresholds": thresholds, "base_config": base_config, "config": asdict(cfg)}, run_out / "robustification_manifest.json")
    print(f"[{timestamp()}] Finished {run_dir.name}; best checkpoint: {run_out / 'checkpoints' / 'best.pt'}", flush=True)


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Robustify BDD100K MoE router with adversarial training and sentinel heads.")
    parser.add_argument("--train-script", type=str, default=DEFAULT_TRAIN_SCRIPT)
    parser.add_argument("--metadata-dir", type=str, default=DEFAULT_METADATA_DIR)
    parser.add_argument("--run-dirs", nargs="*", default=[])
    parser.add_argument("--run-glob", type=str, default=DEFAULT_RUN_GLOB)
    parser.add_argument("--checkpoint-name", type=str, default="best.pt")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-train-samples", type=int, default=4000)
    parser.add_argument("--max-val-samples", type=int, default=1000)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--joint-epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--sentinel-lr-mult", type=float, default=5.0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--attack-eps", type=float, default=0.05)
    parser.add_argument("--attack-steps", type=int, default=5)
    parser.add_argument("--attack-step-size", type=float, default=0.0)
    parser.add_argument("--adv-fraction", type=float, default=0.50)
    parser.add_argument("--clean-accept-target", type=float, default=0.90)
    parser.add_argument("--train-backbone", action="store_true")
    args = parser.parse_args()
    values = vars(args)
    run_dirs = tuple(values.pop("run_dirs"))
    return TrainConfig(run_dirs=run_dirs, **values)


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    ensure_train_import_dependencies()
    train_module = import_module_from_path("bdd100k_moe_train_robustify_module", Path(cfg.train_script).resolve())
    metadata_dir = Path(cfg.metadata_dir).resolve()
    bundle = load_json(metadata_dir / "metadata_bundle.json")
    rows = load_json(metadata_dir / "train.json")
    run_dirs = list_run_dirs(cfg.run_dirs, cfg.run_glob)
    print(f"[{timestamp()}] Robustifying {len(run_dirs)} BDD100K run(s).", flush=True)
    for run_dir in run_dirs:
        train_one_run(cfg, run_dir, train_module, bundle, rows, device)


if __name__ == "__main__":
    main()
