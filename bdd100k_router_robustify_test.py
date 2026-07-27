#!/usr/bin/env python3
"""Evaluate robustified BDD100K MoE routers under router attacks."""

from __future__ import annotations

import argparse
import csv
import importlib.machinery
import importlib.util
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, ImageFile
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

DEFAULT_ROBUSTIFY_SCRIPT = Path(__file__).resolve().with_name("bdd100k_router_robustify_train.py")
DEFAULT_TRAIN_SCRIPT = "/mnt/nvme1n1/bbadjie/SEAS/AAutonomous/Baseline/BDD100k/scripts/bdd100k_moe_train.py"
DEFAULT_TEST_SCRIPT = "/mnt/nvme1n1/bbadjie/SEAS/AAutonomous/Baseline/BDD100k/scripts/bdd100k_moe_test.py"
DEFAULT_METADATA_DIR = "/mnt/nvme1n1/bbadjie/SEAS/AAutonomous/Baseline/BDD100k/data/metadata_files/metadata-New"
DEFAULT_ROBUST_RUN_GLOB = "/mnt/nvme1n1/bbadjie/SEAS/AAutonomous/Adversarial-Robustification/BDD100k/router_sentinel/*"
DEFAULT_OUTPUT_DIR = "/mnt/nvme1n1/bbadjie/SEAS/AAutonomous/Adversarial-Robustification/BDD100k/Add-router_sentinel_eval"
DEFAULT_ATTACKS = ("hsj", "boundary", "square", "nes", "transfer_pgd", "pgd")

EXPERT_INDEX_TO_NAME = {0: "weather", 1: "scene", 2: "time"}
TARGET_OBJECTS = ["car", "pedestrian", "traffic_sign"]
NUM_ROUTER_CLASSES = 3


def timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


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


def install_foolbox_brendel_stub() -> None:
    """Avoid importing Foolbox Brendel-Bethge attacks when numba/NumPy ABI is broken."""

    def _stub_unavailable_attack(name: str):
        class _UnavailableAttack:
            def __init__(self, *args, **kwargs):
                raise RuntimeError(f"{name} is unavailable in this environment. Use hsj or boundary.")

        _UnavailableAttack.__name__ = name
        return _UnavailableAttack

    bb = type(sys)("foolbox.attacks.brendel_bethge")
    bb.__file__ = "<foolbox_brendel_bethge_stub>"
    bb.__package__ = "foolbox.attacks"
    bb.__spec__ = importlib.machinery.ModuleSpec("foolbox.attacks.brendel_bethge", loader=None)
    bb.__getattr__ = lambda name: _stub_unavailable_attack(name)
    bb.BrendelBethgeAttack = _stub_unavailable_attack("BrendelBethgeAttack")
    bb.L0BrendelBethgeAttack = _stub_unavailable_attack("L0BrendelBethgeAttack")
    bb.L1BrendelBethgeAttack = _stub_unavailable_attack("L1BrendelBethgeAttack")
    bb.L2BrendelBethgeAttack = _stub_unavailable_attack("L2BrendelBethgeAttack")
    bb.LinfBrendelBethgeAttack = _stub_unavailable_attack("LinfBrendelBethgeAttack")
    bb.LinfinityBrendelBethgeAttack = bb.LinfBrendelBethgeAttack
    sys.modules["foolbox.attacks.brendel_bethge"] = bb


def list_run_dirs(run_dirs: Sequence[str], run_glob: str) -> List[Path]:
    if run_dirs:
        resolved = [Path(p).expanduser().resolve() for p in run_dirs]
    else:
        glob_path = Path(run_glob).expanduser()
        resolved = sorted(glob_path.parent.glob(glob_path.name)) if glob_path.is_absolute() else sorted(Path(".").glob(run_glob))
        resolved = [p.resolve() for p in resolved]
    resolved = [p for p in resolved if p.exists()]
    if not resolved:
        raise FileNotFoundError(f"No robustified run directories found. Got --robust-run-glob={run_glob!r}")
    return resolved


def make_loader(dataset: Dataset, batch_size: int, num_workers: int, device: torch.device) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )


class DictTensorDataset(Dataset):
    def __init__(self, images: torch.Tensor, source_dataset: Dataset) -> None:
        self.images = images
        self.source_dataset = source_dataset

    def __len__(self) -> int:
        return int(self.images.size(0))

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = dict(self.source_dataset[index])
        item["image"] = self.images[index]
        return item


class BDD100KPixelEvalDataset(Dataset):
    """BDD100K rows with pixel-space images plus all labels needed by eval tables."""

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
            "weather_id": torch.tensor(row["weather_id"], dtype=torch.long),
            "scene_id": torch.tensor(row["scene_id"], dtype=torch.long),
            "time_id": torch.tensor(row["time_id"], dtype=torch.long),
            "weather_mask": torch.tensor(row["weather_mask"], dtype=torch.float32),
            "scene_mask": torch.tensor(row["scene_mask"], dtype=torch.float32),
            "time_mask": torch.tensor(row["time_mask"], dtype=torch.float32),
            "objects": torch.tensor([row["car_present"], row["pedestrian_present"], row["traffic_sign_present"]], dtype=torch.float32),
            "image_path": row["image_path"],
        }


def build_base_model(train_module, bundle: Dict[str, Any], base_config: Dict[str, Any], device: torch.device) -> nn.Module:
    weather_names = [k for k, _ in sorted(bundle["weather_to_id"].items(), key=lambda kv: kv[1])]
    scene_names = [k for k, _ in sorted(bundle["scene_to_id"].items(), key=lambda kv: kv[1])]
    time_names = [k for k, _ in sorted(bundle["time_to_id"].items(), key=lambda kv: kv[1])]
    model = train_module.build_model(
        stage=str(base_config.get("stage", "stage3_moe")),
        backbone_name=str(base_config.get("backbone_name", "convnextv2_tiny.fcmae_ft_in1k")),
        num_weather=len(weather_names),
        num_scene=len(scene_names),
        num_time=len(time_names),
        pretrained=False,
    )
    return model.to(device)


def load_robust_model(
    robust_module,
    train_module,
    bundle: Dict[str, Any],
    robust_run_dir: Path,
    checkpoint_name: str,
    device: torch.device,
) -> Tuple[nn.Module, Dict[str, float], Dict[str, Any]]:
    checkpoint_path = robust_run_dir / "checkpoints" / checkpoint_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing robustified checkpoint: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location=device)
    base_config = payload.get("base_config", {})
    base = build_base_model(train_module, bundle, base_config, device)
    model = robust_module.RobustifiedBDD100KMoE(base).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    return model, payload.get("thresholds", {}), base_config


def clamp_01(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, 0.0, 1.0)


def project_linf(x_adv: torch.Tensor, x_orig: torch.Tensor, eps: float) -> torch.Tensor:
    return clamp_01(torch.max(torch.min(x_adv, x_orig + eps), x_orig - eps))


def router_margin_loss(router_logits: torch.Tensor, y_ref: torch.Tensor) -> torch.Tensor:
    n = router_logits.size(0)
    true = router_logits[torch.arange(n, device=router_logits.device), y_ref]
    other = router_logits.clone()
    other[torch.arange(n, device=router_logits.device), y_ref] = -1e9
    return other.max(dim=1).values - true


def router_logits(router: nn.Module, x: torch.Tensor) -> torch.Tensor:
    return router.routing_only(x) if hasattr(router, "routing_only") else router(x)


def pgd_linf_untargeted(router: nn.Module, x: torch.Tensor, y_ref: torch.Tensor, eps: float, steps: int, step_size: float, random_start: bool) -> torch.Tensor:
    x_orig = x.detach()
    x_adv = project_linf(x_orig + torch.empty_like(x_orig).uniform_(-eps, eps), x_orig, eps) if random_start else x_orig.clone()
    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)
        logits = router_logits(router, x_adv)
        loss = router_margin_loss(logits, y_ref).mean()
        grad = torch.autograd.grad(loss, x_adv, only_inputs=True)[0]
        x_adv = project_linf(x_adv + step_size * torch.sign(grad), x_orig, eps)
    return x_adv.detach()


def pgd_linf_untargeted_restarts(
    router: nn.Module,
    x: torch.Tensor,
    y_ref: torch.Tensor,
    eps: float,
    steps: int,
    step_size: float,
    restarts: int,
) -> torch.Tensor:
    best_adv = x.detach().clone()
    best_loss = torch.full((x.size(0),), -1e9, device=x.device, dtype=x.dtype)
    for restart in range(max(1, restarts)):
        x_adv = pgd_linf_untargeted(router, x, y_ref, eps, steps, step_size, random_start=(restart > 0 or restarts > 1))
        with torch.no_grad():
            loss = router_margin_loss(router_logits(router, x_adv), y_ref)
            improve = loss > best_loss
            best_adv[improve] = x_adv[improve]
            best_loss[improve] = loss[improve]
    return best_adv.detach()


class RoutingOnlyModule(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.routing_only(x)


@torch.no_grad()
def square_attack(router: nn.Module, x: torch.Tensor, y_ref: torch.Tensor, eps: float, queries: int, p_init: float) -> torch.Tensor:
    n, c, h, w = x.shape
    x_orig = x.detach()
    base_pred = torch.argmax(router_logits(router, x_orig), dim=1)
    x_adv = project_linf(x_orig + torch.empty_like(x_orig).uniform_(-eps, eps), x_orig, eps)
    best_loss = router_margin_loss(router_logits(router, x_adv), y_ref)
    success = torch.argmax(router_logits(router, x_adv), dim=1).ne(base_pred)
    for step in range(1, queries):
        active = ~success
        if not bool(active.any()):
            break
        side = max(1, int(round(min(h, w) * max(0.02, p_init * (1.0 - step / max(1, queries))))))
        cand = x_adv.clone()
        for i in torch.where(active)[0].tolist():
            top = int(torch.randint(0, h - side + 1, (1,), device=x.device).item())
            left = int(torch.randint(0, w - side + 1, (1,), device=x.device).item())
            delta = torch.empty((c, side, side), device=x.device).uniform_(-eps, eps)
            cand[i, :, top : top + side, left : left + side] = x_orig[i, :, top : top + side, left : left + side] + delta
        cand = project_linf(cand, x_orig, eps)
        logits = router_logits(router, cand)
        loss = router_margin_loss(logits, y_ref)
        improve = loss > best_loss
        x_adv[improve] = cand[improve]
        best_loss[improve] = loss[improve]
        success = torch.argmax(router_logits(router, x_adv), dim=1).ne(base_pred)
    return x_adv.detach()


@torch.no_grad()
def nes_attack(router: nn.Module, x: torch.Tensor, y_ref: torch.Tensor, eps: float, queries: int, lr: float, sigma: float, samples: int) -> torch.Tensor:
    x_orig = x.detach()
    x_adv = project_linf(x_orig + torch.empty_like(x_orig).uniform_(-eps, eps), x_orig, eps)
    steps = max(1, queries // max(2, 2 * samples))
    for _ in range(steps):
        grad = torch.zeros_like(x_adv)
        for _sample in range(samples):
            u = torch.randn_like(x_adv)
            loss_plus = router_margin_loss(router_logits(router, project_linf(x_adv + sigma * u, x_orig, eps)), y_ref)
            loss_minus = router_margin_loss(router_logits(router, project_linf(x_adv - sigma * u, x_orig, eps)), y_ref)
            grad += ((loss_plus - loss_minus) / (2.0 * sigma)).view(-1, 1, 1, 1) * u
        grad = grad / float(samples)
        x_adv = project_linf(x_adv + lr * torch.sign(grad), x_orig, eps)
    return x_adv.detach()


class SurrogateRouterSmall(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(128 * 4 * 4, 256), nn.ReLU(inplace=True), nn.Dropout(0.25), nn.Linear(256, NUM_ROUTER_CLASSES))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class PseudoLabelDataset(Dataset):
    def __init__(self, base: Dataset, labels: torch.Tensor) -> None:
        self.base = base
        self.labels = labels.cpu().long()

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.base[index]["image"], self.labels[index]


def train_surrogate(surrogate: nn.Module, dataset: Dataset, device: torch.device, epochs: int, batch_size: int, num_workers: int, lr: float) -> None:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=(device.type == "cuda"), persistent_workers=(num_workers > 0))
    optimizer = optim.AdamW(surrogate.parameters(), lr=lr, weight_decay=1e-4)
    surrogate.train()
    for epoch in range(1, epochs + 1):
        correct = 0
        total = 0
        losses = []
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = surrogate(images)
            loss = nn.functional.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
            correct += int(torch.argmax(logits.detach(), dim=1).eq(labels).sum().item())
            total += int(labels.numel())
        print(f"[{timestamp()}] surrogate epoch {epoch}/{epochs}: loss={np.mean(losses):.5f}, acc={correct / max(1, total):.4f}", flush=True)
    surrogate.eval()


@torch.no_grad()
def collect_clean_labels(model: nn.Module, loader: DataLoader, device: torch.device) -> torch.Tensor:
    labels: List[torch.Tensor] = []
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        labels.append(torch.argmax(model.routing_only(x), dim=1).cpu())
    return torch.cat(labels, dim=0).long()


def craft_attack_images(model: nn.Module, loader: DataLoader, labels: torch.Tensor, args, device: torch.device, source_dataset: Dataset) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
    adv_chunks: List[torch.Tensor] = []
    success_chunks: List[np.ndarray] = []
    initialized_chunks: List[np.ndarray] = []
    attack = args.current_attack.lower()
    step_size = args.pgd_step_size if args.pgd_step_size > 0 else (args.eps / max(1, args.pgd_steps)) * 2.5

    surrogate: Optional[nn.Module] = None
    if attack == "transfer_pgd":
        surrogate = SurrogateRouterSmall().to(device)
        train_surrogate(surrogate, PseudoLabelDataset(source_dataset, labels), device, args.sur_epochs, args.sur_batch_size, args.num_workers, args.sur_lr)

    if attack in {"hsj", "boundary"}:
        try:
            install_foolbox_brendel_stub()
            import foolbox as fb
        except ImportError as exc:
            raise RuntimeError("Foolbox is required for hsj/boundary decision attacks.") from exc
        fmodel = fb.PyTorchModel(RoutingOnlyModule(model).to(device).eval(), bounds=(0.0, 1.0), device=device)
        if attack == "hsj":
            attacker = fb.attacks.HopSkipJumpAttack(
                steps=args.decision_steps,
                initial_gradient_eval_steps=args.hsj_initial_gradient_eval_steps,
                max_gradient_eval_steps=args.hsj_max_gradient_eval_steps,
                constraint=args.hsj_constraint,
            )
        else:
            attacker = fb.attacks.BoundaryAttack(steps=args.decision_steps)

        def run_decision_attack(xb: torch.Tensor, yb: torch.Tensor) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
            start = pgd_linf_untargeted_restarts(
                model,
                xb,
                yb,
                eps=args.decision_init_eps if args.decision_init_eps > 0 else args.eps,
                steps=args.decision_init_pgd_steps,
                step_size=args.decision_init_pgd_step_size if args.decision_init_pgd_step_size > 0 else ((args.decision_init_eps if args.decision_init_eps > 0 else args.eps) / max(1, args.decision_init_pgd_steps)) * 2.5,
                restarts=args.decision_init_restarts,
            )
            with torch.no_grad():
                start_success = torch.argmax(model.routing_only(start), dim=1).ne(yb).detach().cpu().numpy().astype(np.bool_)
            clipped_batch = xb.detach().clone()
            success_batch = np.zeros((xb.shape[0],), dtype=np.bool_)
            initialized_batch = start_success.reshape(-1).copy()
            failed = 0
            oom_failed = 0
            for i in range(xb.shape[0]):
                try:
                    kwargs = {"starting_points": start[i : i + 1]} if initialized_batch[i] else {}
                    _raw_i, clipped_i, success_i = attacker(fmodel, xb[i : i + 1], yb[i : i + 1], epsilons=args.eps, **kwargs)
                    initialized_batch[i] = True
                    clipped_batch[i : i + 1] = clipped_i.detach()
                    try:
                        success_i_np = success_i.detach().cpu().numpy().astype(np.bool_)
                    except Exception:
                        success_i_np = np.asarray(success_i, dtype=np.bool_)
                    success_batch[i] = bool(success_i_np.reshape(-1)[0])
                except ValueError as exc:
                    if "init_attack failed" not in str(exc):
                        raise
                    failed += 1
                except (torch.OutOfMemoryError, RuntimeError) as exc:
                    if isinstance(exc, RuntimeError) and "out of memory" not in str(exc).lower():
                        raise
                    oom_failed += 1
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
            if failed:
                print(f"[{timestamp()}] Warning: {attack} init_attack failed for {failed}/{xb.shape[0]} sample(s) after PGD-start fallback; using clean images for those samples.", flush=True)
            if oom_failed:
                print(f"[{timestamp()}] Warning: {attack} ran out of GPU memory for {oom_failed}/{xb.shape[0]} sample(s); using clean images for those samples. Reduce --hsj-initial-gradient-eval-steps/--hsj-max-gradient-eval-steps if this persists.", flush=True)
            return clipped_batch, success_batch, initialized_batch

    offset = 0
    for batch_idx, batch in enumerate(loader, start=1):
        x = batch["image"].to(device, non_blocking=True)
        n = int(x.size(0))
        y = labels[offset : offset + n].to(device, non_blocking=True)
        offset += n
        print(f"[{timestamp()}] {attack}: batch {batch_idx}/{len(loader)} ({offset}/{len(loader.dataset)})", flush=True)
        if attack == "pgd":
            x_adv = pgd_linf_untargeted_restarts(model, x, y, args.eps, args.pgd_steps, step_size, args.pgd_restarts)
        elif attack == "transfer_pgd":
            assert surrogate is not None
            x_adv = pgd_linf_untargeted_restarts(surrogate, x, y, args.eps, args.pgd_steps, step_size, args.pgd_restarts)
        elif attack == "square":
            x_adv = square_attack(model, x, y, args.eps, args.queries, args.square_p_init)
        elif attack == "nes":
            x_adv = nes_attack(model, x, y, args.eps, args.queries, args.nes_lr, args.nes_sigma, args.nes_samples)
        elif attack in {"hsj", "boundary"}:
            x_adv, success_np, initialized_np = run_decision_attack(x, y)
        else:
            raise ValueError(f"Unknown attack: {args.attack}")
        with torch.no_grad():
            success = torch.as_tensor(success_np, device=device, dtype=torch.bool) if attack in {"hsj", "boundary"} else torch.argmax(model.routing_only(x_adv), dim=1).ne(y)
            if attack not in {"hsj", "boundary"}:
                initialized_np = np.ones((n,), dtype=np.bool_)
            initialized = torch.as_tensor(initialized_np, device=device, dtype=torch.bool) if attack in {"hsj", "boundary"} else torch.ones_like(success, dtype=torch.bool)
        adv_chunks.append(x_adv.detach().cpu())
        success_chunks.append(success.cpu().numpy().astype(np.bool_))
        initialized_chunks.append(initialized.cpu().numpy().astype(np.bool_))
    return torch.cat(adv_chunks, dim=0), np.concatenate(success_chunks, axis=0), np.concatenate(initialized_chunks, axis=0)


@torch.no_grad()
def collect_predictions(model: nn.Module, loader: DataLoader, device: torch.device, thresholds: Dict[str, float]) -> Dict[str, Any]:
    chunk_keys = [
        "y_true",
        "weather_id",
        "scene_id",
        "time_id",
        "weather_mask",
        "scene_mask",
        "time_mask",
        "probs_fused",
        "probs_weather",
        "probs_scene",
        "probs_time",
        "weather_logits",
        "scene_logits",
        "time_logits",
        "alpha",
        "router_logits",
        "risk",
        "attack_prob",
        "action_id",
    ]
    chunks: Dict[str, List[np.ndarray]] = {k: [] for k in chunk_keys}
    image_paths: List[str] = []
    tau_allow = float(thresholds.get("allow", 1e9))
    tau_review = float(thresholds.get("review", thresholds.get("reject", 1e9)))
    tau_reject = float(thresholds.get("reject", 1e9))
    for batch in loader:
        image_paths.extend(batch.get("image_path", [""] * int(batch["image"].size(0))))
        x = batch["image"].to(device, non_blocking=True)
        out = model(x, audit=True, calibrated_attack=True)
        risk = out["risk"]
        action = torch.zeros_like(risk, dtype=torch.long)
        action[(risk >= tau_allow) & (risk < tau_review)] = 1
        action[(risk >= tau_review) & (risk < tau_reject)] = 2
        action[risk >= tau_reject] = 3
        chunks["y_true"].append(batch["objects"].numpy())
        for key in ["weather_id", "scene_id", "time_id", "weather_mask", "scene_mask", "time_mask"]:
            if key in batch:
                chunks[key].append(batch[key].numpy())
        chunks["probs_fused"].append(torch.sigmoid(out["obj_fused_logits"]).cpu().numpy())
        chunks["probs_weather"].append(torch.sigmoid(out["obj_w_logits"]).cpu().numpy())
        chunks["probs_scene"].append(torch.sigmoid(out["obj_s_logits"]).cpu().numpy())
        chunks["probs_time"].append(torch.sigmoid(out["obj_t_logits"]).cpu().numpy())
        chunks["weather_logits"].append(out["weather_logits"].cpu().numpy())
        chunks["scene_logits"].append(out["scene_logits"].cpu().numpy())
        chunks["time_logits"].append(out["time_logits"].cpu().numpy())
        chunks["alpha"].append(out["alpha"].cpu().numpy() if "alpha" in out else torch.softmax(out["router_logits"], dim=1).cpu().numpy())
        chunks["router_logits"].append(out["router_logits"].cpu().numpy())
        chunks["risk"].append(risk.cpu().numpy())
        chunks["attack_prob"].append(torch.softmax(out["attack_logits"], dim=1)[:, 1].cpu().numpy())
        chunks["action_id"].append(action.cpu().numpy())
    preds = {k: (np.concatenate(v, axis=0) if v else np.empty((0,), dtype=np.float32)) for k, v in chunks.items()}
    preds["probs"] = preds["probs_fused"]
    preds["image_path"] = image_paths
    preds["inference_ms_per_image"] = float("nan")
    return preds


def safe_metric(fn, *args, **kwargs) -> float:
    try:
        value = fn(*args, **kwargs)
        return float(np.mean(value)) if isinstance(value, np.ndarray) else float(value)
    except Exception:
        return float("nan")


def multilabel_summary(prefix: str, y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(np.int64)
    ap_scores = [safe_metric(average_precision_score, y_true[:, i], y_prob[:, i]) for i in range(y_true.shape[1]) if np.sum(y_true[:, i]) > 0]
    auc_scores = [safe_metric(roc_auc_score, y_true[:, i], y_prob[:, i]) for i in range(y_true.shape[1]) if len(np.unique(y_true[:, i])) > 1]
    return {
        "name": prefix,
        "micro_f1": safe_metric(f1_score, y_true, y_pred, average="micro", zero_division=0),
        "macro_f1": safe_metric(f1_score, y_true, y_pred, average="macro", zero_division=0),
        "mAP": float(np.mean(ap_scores)) if ap_scores else float("nan"),
        "macro_auroc": float(np.mean(auc_scores)) if auc_scores else float("nan"),
    }


def router_reference_margins(alpha: np.ndarray, reference_top1: np.ndarray) -> np.ndarray:
    out = np.empty((alpha.shape[0],), dtype=np.float64)
    for i, ref in enumerate(reference_top1.astype(np.int64)):
        out[i] = float(alpha[i, ref] - np.max(np.delete(alpha[i], ref)))
    return out


def top1_top2_margin_np(logits: np.ndarray) -> np.ndarray:
    top2 = np.sort(logits, axis=1)[:, -2:]
    return (top2[:, 1] - top2[:, 0]).astype(np.float64)


def routing_recall_from_router_probs(router_probs: np.ndarray, y_ref: np.ndarray, topk: int) -> float:
    topk = int(max(1, min(topk, router_probs.shape[1])))
    topk_idx = np.argpartition(-router_probs, kth=topk - 1, axis=1)[:, :topk]
    return float(np.mean([int(y_ref[i]) in set(topk_idx[i].tolist()) for i in range(router_probs.shape[0])]))


def magnitude_overall_route_failure(clean_probs: np.ndarray, adv_probs: np.ndarray, y_ref: np.ndarray, topk: int) -> float:
    topk = int(max(1, min(topk, clean_probs.shape[1])))
    clean_topk = np.argpartition(-clean_probs, kth=topk - 1, axis=1)[:, :topk]
    adv_topk = np.argpartition(-adv_probs, kth=topk - 1, axis=1)[:, :topk]
    failures = [int(y_ref[i]) in set(clean_topk[i].tolist()) and int(y_ref[i]) not in set(adv_topk[i].tolist()) for i in range(len(y_ref))]
    return float(np.mean(failures))


def magnitude_each_expert_failure_mean(clean_probs: np.ndarray, adv_probs: np.ndarray, y_ref: np.ndarray) -> float:
    drops = []
    for c in range(clean_probs.shape[1]):
        mask = y_ref.astype(np.int64) == c
        if np.any(mask):
            drops.append(float(np.mean(clean_probs[mask, c] - adv_probs[mask, c])))
    return float(np.mean(drops)) if drops else float("nan")


def fused_margin_from_probs(probs: np.ndarray) -> np.ndarray:
    if probs.shape[1] < 2:
        return probs.max(axis=1).astype(np.float64)
    top2 = np.sort(probs, axis=1)[:, -2:]
    return (top2[:, 1] - top2[:, 0]).astype(np.float64)


def positive_label_score_degradation(clean_probs: np.ndarray, adv_probs: np.ndarray, y_true: np.ndarray) -> float:
    mask = y_true.astype(bool)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(clean_probs[mask] - adv_probs[mask]))


def multilabel_exact_match_mask(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> np.ndarray:
    return np.all((y_prob >= threshold).astype(np.int64) == y_true.astype(np.int64), axis=1)


def binary_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.int64)
    y_score = y_score.astype(np.float64)
    pos = int((y_true == 1).sum())
    neg = int((y_true == 0).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    tp = np.cumsum(y_sorted == 1) / pos
    fp = np.cumsum(y_sorted == 0) / neg
    return float(np.trapezoid(np.concatenate([[0.0], tp, [1.0]]), np.concatenate([[0.0], fp, [1.0]])))


def coverage_vs_robustness_curve(adv: Dict[str, Any], threshold: float, n_points: int = 25) -> List[Dict[str, Any]]:
    risk = adv["risk"].astype(np.float64)
    rows: List[Dict[str, Any]] = []
    if risk.size == 0:
        return rows
    for tau in np.linspace(float(np.min(risk)), float(np.max(risk)), n_points):
        accept = risk < tau
        metrics = (
            multilabel_summary("fused_moe_accepted", adv["y_true"][accept], adv["probs_fused"][accept], threshold)
            if np.any(accept)
            else {
                "name": "fused_moe_accepted",
                "micro_f1": float("nan"),
                "macro_f1": float("nan"),
                "mAP": float("nan"),
                "macro_auroc": float("nan"),
            }
        )
        rows.append(
            {
                "risk_threshold": float(tau),
                "coverage": float(np.mean(accept)),
                **metrics,
            }
        )
    return rows


def compute_after_defense_summary(
    clean: Dict[str, Any],
    adv: Dict[str, Any],
    success: np.ndarray,
    initialized: np.ndarray,
    seed: int,
    run_dir: Path,
    attack: str,
    threshold: float,
    topk: int,
) -> List[Dict[str, Any]]:
    clean_top1 = np.argmax(clean["alpha"], axis=1)
    adv_top1 = np.argmax(adv["alpha"], axis=1)
    allow = adv["action_id"] == 0
    cautious = adv["action_id"] == 1
    accepted = allow | cautious
    review = adv["action_id"] == 2
    reject = adv["action_id"] == 3
    blocked = review | reject
    clean_margin = router_reference_margins(clean["alpha"], clean_top1)
    adv_margin = router_reference_margins(adv["alpha"], clean_top1)
    clean_fused_margin = fused_margin_from_probs(clean["probs_fused"])
    adv_fused_margin = fused_margin_from_probs(adv["probs_fused"])
    clean_fused = multilabel_summary("fused_moe_clean", clean["y_true"], clean["probs_fused"], threshold)
    adv_all = multilabel_summary("fused_moe_adv_all", adv["y_true"], adv["probs_fused"], threshold)
    adv_acc = multilabel_summary("fused_moe_adv_accepted", adv["y_true"][accepted], adv["probs_fused"][accepted], threshold) if np.any(accepted) else {}
    robust_accuracy = adv_acc.get("micro_f1", float("nan"))
    attack_success_rate = float(np.mean(accepted & success.astype(bool))) if success.size else float("nan")
    expert_score_deg = positive_label_score_degradation(clean["probs_fused"], adv["probs_fused"], clean["y_true"])
    exact_correct = multilabel_exact_match_mask(adv["y_true"], adv["probs_fused"], threshold)
    y_det_true = np.concatenate([np.zeros_like(clean["attack_prob"], dtype=np.int64), np.ones_like(adv["attack_prob"], dtype=np.int64)])
    y_det_score = np.concatenate([clean["attack_prob"], adv["attack_prob"]])
    return [
        {
            "seed": seed,
            "run_dir": str(run_dir),
            "attack": attack,
            "topk": int(topk),
            "num_samples": int(len(success)),
            "attack_initialization_success_rate": float(np.mean(initialized)) if initialized.size else float("nan"),
            "attack_initialization_failure_rate": float(np.mean(~initialized)) if initialized.size else float("nan"),
            "router_attack_success_rate": float(np.mean(success)) if success.size else float("nan"),
            "router_attack_success_rate_initialized_only": float(np.mean(success[initialized])) if np.any(initialized) else float("nan"),
            "router_top1_flip_rate": float(np.mean(clean_top1 != adv_top1)),
            "routing_recall_under_attack": routing_recall_from_router_probs(adv["alpha"], clean_top1, topk),
            "robust_accuracy": robust_accuracy,
            "attack_success_rate": attack_success_rate,
            "attack_success_rate_after_defense": attack_success_rate,
            "attack_success_rate_after_defense_initialized_only": float(np.mean((accepted & success.astype(bool))[initialized])) if np.any(initialized) else float("nan"),
            "sentinel_allow_rate": float(np.mean(allow)),
            "cautious_route_rate": float(np.mean(cautious)),
            "sentinel_accept_rate": float(np.mean(accepted)),
            "review_rate": float(np.mean(review)),
            "reject_rate": float(np.mean(reject)),
            "adv_block_rate": float(np.mean(blocked)),
            "sentinel_block_given_attack_success": float(np.mean(blocked[success])) if np.any(success) else float("nan"),
            "adversarial_detection_auroc": binary_auroc(y_det_true, y_det_score),
            "clean_micro_f1": clean_fused["micro_f1"],
            "adv_all_micro_f1": adv_all["micro_f1"],
            "adv_accepted_micro_f1": adv_acc.get("micro_f1", float("nan")),
            "clean_mAP": clean_fused["mAP"],
            "adv_all_mAP": adv_all["mAP"],
            "adv_accepted_mAP": adv_acc.get("mAP", float("nan")),
            "fused_micro_f1_degradation": float(clean_fused["micro_f1"] - adv_all["micro_f1"]),
            "fused_mAP_degradation": float(clean_fused["mAP"] - adv_all["mAP"]),
            "fused_margin_degradation": float(np.mean(clean_fused_margin - adv_fused_margin)),
            "gate_margin_degradation": float(np.mean(clean_margin - adv_margin)),
            "expert_conditional_score_degradation": expert_score_deg,
            "magnitude_overall_route_failure": magnitude_overall_route_failure(clean["alpha"], adv["alpha"], clean_top1, topk),
            "magnitude_each_expert_failure_mean": magnitude_each_expert_failure_mean(clean["alpha"], adv["alpha"], clean_top1),
            "magnitude_overall_fused_failure": expert_score_deg,
            "prevented_misrouting_rate": float(np.mean(blocked[adv_top1 != clean_top1])) if np.any(adv_top1 != clean_top1) else float("nan"),
            "robust_accuracy_on_accepted_samples": robust_accuracy,
            "robust_accuracy_with_review_treated_as_safe_abstention": float(np.mean(exact_correct | review)),
            "robust_accuracy_with_block_treated_as_safe": float(np.mean(exact_correct | blocked)),
            "mean_abs_fused_prob_shift": float(np.mean(np.abs(clean["probs_fused"] - adv["probs_fused"]))),
            "clean_mean_risk": float(np.mean(clean["risk"])),
            "adv_mean_risk": float(np.mean(adv["risk"])),
            "clean_mean_attack_prob": float(np.mean(clean["attack_prob"])),
            "adv_mean_attack_prob": float(np.mean(adv["attack_prob"])),
        }
    ]


def add_run_info(rows: List[Dict[str, Any]], seed: int, run_dir: Path, attack: str, split: str) -> None:
    for row in rows:
        row["seed"] = seed
        row["run_dir"] = str(run_dir)
        row["attack"] = attack
        row["split"] = split


def compute_eval_tables(
    preds: Dict[str, Any],
    eval_module,
    weather_names: Sequence[str],
    scene_names: Sequence[str],
    time_names: Sequence[str],
    threshold: float,
    seed: int,
    run_dir: Path,
    attack: str,
    split: str,
) -> Dict[str, List[Dict[str, Any]]]:
    fused_summary = eval_module.compute_multilabel_metrics("fused_moe", preds["y_true"], preds["probs_fused"], threshold)
    fused_summary["inference_ms"] = preds["inference_ms_per_image"]
    tables = {
        "fused_summary": [fused_summary],
        "fused_per_class": eval_module.compute_per_class_object_metrics("fused_moe", preds["y_true"], preds["probs_fused"], threshold),
        "expert_per_class": [],
        "expert_specialist": eval_module.compute_specialist_metrics(preds, weather_names, scene_names, time_names, threshold),
        "condition": [],
        "router_per_class": eval_module.compute_router_per_class_rows(preds),
        "router_condition": eval_module.compute_router_condition_rows(preds, weather_names, scene_names, time_names),
    }
    for head_name, probs in [
        ("weather_expert", preds["probs_weather"]),
        ("scene_expert", preds["probs_scene"]),
        ("time_expert", preds["probs_time"]),
    ]:
        tables["expert_per_class"].extend(eval_module.compute_per_class_object_metrics(head_name, preds["y_true"], probs, threshold))
    for task, logits, y_true, mask, names in [
        ("weather", preds["weather_logits"], preds["weather_id"], preds["weather_mask"], weather_names),
        ("scene", preds["scene_logits"], preds["scene_id"], preds["scene_mask"], scene_names),
        ("time", preds["time_logits"], preds["time_id"], preds["time_mask"], time_names),
    ]:
        row, _ = eval_module.compute_condition_metrics(task, logits, y_true, mask, names)
        tables["condition"].append(row)
    router_summary, _ = eval_module.compute_router_metrics(preds, weather_names, scene_names, time_names)
    tables["router_summary"] = router_summary
    for table_rows in tables.values():
        add_run_info(table_rows, seed, run_dir, attack, split)
    return tables


def routing_flip_rows(clean: Dict[str, Any], adv: Dict[str, Any], success: np.ndarray, seed: int, run_dir: Path, attack: str) -> List[Dict[str, Any]]:
    clean_top1 = np.argmax(clean["alpha"], axis=1)
    adv_top1 = np.argmax(adv["alpha"], axis=1)
    blocked = adv["action_id"] >= 2
    accepted = ~blocked
    rows = [
        {
            "seed": seed,
            "run_dir": str(run_dir),
            "attack": attack,
            "subset": "all",
            "n": int(clean_top1.size),
            "attack_success_rate_router": float(np.mean(success)) if success.size else float("nan"),
            "router_top1_flip_rate": float(np.mean(clean_top1 != adv_top1)),
            "router_mean_l1_alpha_shift": float(np.mean(np.abs(clean["alpha"] - adv["alpha"]).sum(axis=1))),
            "router_mean_l2_alpha_shift": float(np.mean(np.linalg.norm(clean["alpha"] - adv["alpha"], axis=1))),
            "fused_mean_abs_prob_shift": float(np.mean(np.abs(clean["probs_fused"] - adv["probs_fused"]))),
            "sentinel_block_rate": float(np.mean(blocked)),
            "sentinel_allow_rate": float(np.mean(accepted)),
            "cautious_route_rate": float(np.mean(adv["action_id"] == 1)),
            "sentinel_block_given_attack_success": float(np.mean(blocked[success])) if np.any(success) else float("nan"),
        }
    ]
    for i, name in EXPERT_INDEX_TO_NAME.items():
        mask = clean_top1 == i
        if not np.any(mask):
            continue
        rows.append(
            {
                "seed": seed,
                "run_dir": str(run_dir),
                "attack": attack,
                "subset": f"clean_top1_{name}",
                "n": int(mask.sum()),
                "attack_success_rate_router": float(np.mean(success[mask])),
                "router_top1_flip_rate": float(np.mean(clean_top1[mask] != adv_top1[mask])),
                "router_mean_l1_alpha_shift": float(np.mean(np.abs(clean["alpha"][mask] - adv["alpha"][mask]).sum(axis=1))),
                "router_mean_l2_alpha_shift": float(np.mean(np.linalg.norm(clean["alpha"][mask] - adv["alpha"][mask], axis=1))),
                "fused_mean_abs_prob_shift": float(np.mean(np.abs(clean["probs_fused"][mask] - adv["probs_fused"][mask]))),
                "sentinel_block_rate": float(np.mean(blocked[mask])),
                "sentinel_allow_rate": float(np.mean(accepted[mask])),
                "cautious_route_rate": float(np.mean((adv["action_id"] == 1)[mask])),
                "sentinel_block_given_attack_success": float(np.mean(blocked[mask][success[mask]])) if np.any(success[mask]) else float("nan"),
            }
        )
    return rows


def degradation_rows(clean_tables: Dict[str, List[Dict[str, Any]]], adv_tables: Dict[str, List[Dict[str, Any]]], seed: int, run_dir: Path, attack: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    metric_names = ["micro_f1", "macro_f1", "accuracy", "balanced_accuracy", "mAP", "micro_mAP", "macro_auroc", "micro_auroc", "ece", "brier"]
    clean_lookup = {row["name"]: row for row in clean_tables["fused_summary"]}
    adv_lookup = {row["name"]: row for row in adv_tables["fused_summary"]}
    for name, c_row in clean_lookup.items():
        a_row = adv_lookup.get(name)
        if a_row is None:
            continue
        for metric in metric_names:
            if metric in c_row and metric in a_row:
                rows.append(
                    {
                        "seed": seed,
                        "run_dir": str(run_dir),
                        "attack": attack,
                        "component": name,
                        "metric": metric,
                        "clean": c_row[metric],
                        "adv": a_row[metric],
                        "delta_adv_minus_clean": float(a_row[metric]) - float(c_row[metric]),
                    }
                )
    return rows


def positive_object_fused_probability_drops(clean_probs: np.ndarray, adv_probs: np.ndarray, y_true: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    all_drops: List[np.ndarray] = []
    by_object: Dict[str, np.ndarray] = {}
    for obj_idx, obj_name in enumerate(TARGET_OBJECTS):
        mask = y_true[:, obj_idx] == 1
        drops = (clean_probs[mask, obj_idx] - adv_probs[mask, obj_idx]).astype(np.float64)
        by_object[obj_name] = drops
        if drops.size:
            all_drops.append(drops)
    return (np.concatenate(all_drops, axis=0) if all_drops else np.empty((0,), dtype=np.float64)), by_object


def finite_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values[np.isfinite(values)]


def plot_hist(path: Path, values: np.ndarray, title: str, xlabel: str, color: str) -> None:
    values = finite_values(values)
    if values.size == 0:
        return
    import matplotlib.pyplot as plt

    ensure_dir(path.parent)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.hist(values, bins=50, density=True, color=color, edgecolor="black", linewidth=0.4)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.2)
    ax.set_title(title, fontsize=22, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=21, fontweight="bold", labelpad=18)
    ax.set_ylabel("Probability density", fontsize=21, fontweight="bold")
    ax.tick_params(axis="both", labelsize=20)
    for tick_label in ax.get_xticklabels():
        tick_label.set_rotation(45)
        tick_label.set_ha("right")
        tick_label.set_rotation_mode("anchor")
        tick_label.set_fontweight("bold")
    for tick_label in ax.get_yticklabels():
        tick_label.set_fontweight("bold")
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.24)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def margin_plot_rows_and_files(clean: Dict[str, Any], adv: Dict[str, Any], plots_dir: Path, seed: int, run_dir: Path, attack: str) -> List[Dict[str, Any]]:
    run_plot_dir = plots_dir / f"{run_dir.name}_seed_{seed}"
    clean_top1 = np.argmax(clean["alpha"], axis=1)
    router_margin_clean = router_reference_margins(clean["alpha"], clean_top1)
    router_margin_adv = router_reference_margins(adv["alpha"], clean_top1)
    router_margin_degradation = router_margin_clean - router_margin_adv
    fused_margin_clean = fused_margin_from_probs(clean["probs_fused"])
    fused_margin_adv = fused_margin_from_probs(adv["probs_fused"])
    fused_drop_all, fused_drop_by_object = positive_object_fused_probability_drops(clean["probs_fused"], adv["probs_fused"], clean["y_true"])

    plot_hist(run_plot_dir / f"gate_margin_clean_{attack}.png", router_margin_clean, "Gate Margin (Robustified Clean)", "Clean gate margin", "green")
    plot_hist(run_plot_dir / f"gate_margin_adv_{attack}.png", router_margin_adv, f"Gate Margin Adv ({attack})", "Gate margin (adversarial)", "orange")
    plot_hist(run_plot_dir / f"fused_margin_clean_{attack}.png", fused_margin_clean, f"Fused Margin Distribution Clean ({attack})", "Fused margin", "green")
    plot_hist(run_plot_dir / f"fused_margin_adv_{attack}.png", fused_margin_adv, f"Fused Margin Distribution Adv. ({attack})", "Fused margin", "orange")
    plot_hist(run_plot_dir / f"gate_margin_degradation_{attack}.png", router_margin_degradation, f"Gate Margin Degradation ({attack})", "Gate margin drop after attack", "crimson")
    plot_hist(run_plot_dir / f"fused_positive_object_prob_drop_{attack}.png", fused_drop_all, f"FPoPD ({attack})", "Object probability drop after attack", "crimson")
    for obj_name, drops in fused_drop_by_object.items():
        plot_hist(run_plot_dir / f"fused_positive_object_prob_drop_{obj_name}_{attack}.png", drops, f"FPoPD - {obj_name} ({attack})", "Object probability drop after attack", "crimson")

    specs = [
        ("router_margin_clean", router_margin_clean),
        ("router_margin_adv", router_margin_adv),
        ("fused_margin_clean", fused_margin_clean),
        ("fused_margin_adv", fused_margin_adv),
        ("router_margin_degradation", router_margin_degradation),
        ("fused_positive_object_prob_drop_all", fused_drop_all),
    ]
    specs.extend((f"fused_positive_object_prob_drop_{obj_name}", drops) for obj_name, drops in fused_drop_by_object.items())
    rows = []
    for name, values in specs:
        values = finite_values(values)
        rows.append(
            {
                "seed": seed,
                "run_dir": str(run_dir),
                "attack": attack,
                "name": name,
                "n": int(values.size),
                "mean": float(np.mean(values)) if values.size else float("nan"),
                "median": float(np.median(values)) if values.size else float("nan"),
                "std": float(np.std(values)) if values.size else float("nan"),
            }
        )
    return rows


def action_name(action_id: int) -> str:
    return ["allow", "cautious_route", "review", "reject"][int(action_id)]


def object_label_string(bits: np.ndarray) -> str:
    labels = [TARGET_OBJECTS[i] for i, bit in enumerate(bits.astype(np.int64).tolist()) if bit == 1]
    return "|".join(labels) if labels else "none"


def build_actions_rows(
    preds: Dict[str, Any],
    seed: int,
    run_dir: Path,
    attack: str,
    threshold: float,
    success: Optional[np.ndarray] = None,
    initialized: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    router_top1 = np.argmax(preds["alpha"], axis=1)
    router_margin = router_reference_margins(preds["alpha"], router_top1)
    fused_margin = fused_margin_from_probs(preds["probs_fused"])
    y_pred = (preds["probs_fused"] >= threshold).astype(np.int64)
    rows: List[Dict[str, Any]] = []
    for i, image_path in enumerate(preds["image_path"]):
        action_id = int(preds["action_id"][i])
        row: Dict[str, Any] = {
            "sample_id": i,
            "run": run_dir.name,
            "run_dir": str(run_dir),
            "seed": seed,
            "attack": attack,
            "image_path": image_path,
            "y_true": object_label_string(preds["y_true"][i]),
            "gate_pred": int(router_top1[i]),
            "gate_pred_name": EXPERT_INDEX_TO_NAME[int(router_top1[i])],
            "defended_pred": object_label_string(y_pred[i]) if action_id < 2 else "-1",
            "risk": float(preds["risk"][i]),
            "attack_prob": float(preds["attack_prob"][i]),
            "action_id": action_id,
            "action_name": action_name(action_id),
            "gate_margin": float(router_margin[i]),
            "fused_margin": float(fused_margin[i]),
        }
        for obj_idx, obj_name in enumerate(TARGET_OBJECTS):
            row[f"true_{obj_name}"] = int(preds["y_true"][i, obj_idx])
            row[f"pred_{obj_name}"] = int(y_pred[i, obj_idx])
            row[f"prob_{obj_name}"] = float(preds["probs_fused"][i, obj_idx])
        for expert_idx, expert_name in EXPERT_INDEX_TO_NAME.items():
            row[f"gate_prob_{expert_name}"] = float(preds["alpha"][i, expert_idx])
        if success is not None:
            row["attack_succeeded_on_gate"] = int(success[i])
        if initialized is not None:
            row["attack_initialized"] = int(initialized[i])
        rows.append(row)
    return rows


def aggregate_and_write(eval_module, csv_dir: Path, prefix: str, rows: List[Dict[str, Any]], key_cols: Sequence[str]) -> None:
    eval_module.write_csv(csv_dir / f"{prefix}_per_run.csv", rows)
    eval_module.write_csv(csv_dir / f"{prefix}_aggregate.csv", eval_module.summarize_table(rows, key_cols=key_cols))


def summarize_run(clean: Dict[str, Any], adv: Dict[str, Any], success: np.ndarray, threshold: float, run_name: str, attack: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    clean_top1 = np.argmax(clean["alpha"], axis=1)
    adv_top1 = np.argmax(adv["alpha"], axis=1)
    blocked = adv["action_id"] >= 2
    accepted = ~blocked
    rows = [
        {
            "run": run_name,
            "attack": attack,
            "n": int(len(success)),
            "router_attack_success_rate": float(np.mean(success)),
            "router_top1_flip_rate": float(np.mean(clean_top1 != adv_top1)),
            "sentinel_block_rate": float(np.mean(blocked)),
            "sentinel_allow_rate": float(np.mean(adv["action_id"] == 0)),
            "cautious_route_rate": float(np.mean(adv["action_id"] == 1)),
            "sentinel_accept_rate": float(np.mean(accepted)),
            "sentinel_block_given_attack_success": float(np.mean(blocked[success])) if np.any(success) else float("nan"),
            "clean_mean_router_margin": float(np.mean(router_reference_margins(clean["alpha"], clean_top1))),
            "adv_mean_router_margin_vs_clean_top1": float(np.mean(router_reference_margins(adv["alpha"], clean_top1))),
            "mean_abs_fused_prob_shift": float(np.mean(np.abs(clean["probs"] - adv["probs"]))),
        }
    ]
    metric_rows = [
        {"run": run_name, "attack": attack, "split": "clean", **multilabel_summary("fused_moe", clean["y_true"], clean["probs"], threshold)},
        {"run": run_name, "attack": attack, "split": "adv_all", **multilabel_summary("fused_moe", adv["y_true"], adv["probs"], threshold)},
    ]
    if np.any(accepted):
        metric_rows.append({"run": run_name, "attack": attack, "split": "adv_accepted_only", **multilabel_summary("fused_moe", adv["y_true"][accepted], adv["probs"][accepted], threshold)})
    return rows, metric_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate BDD100K robustified router under attacks.")
    parser.add_argument("--robustify-script", type=str, default=str(DEFAULT_ROBUSTIFY_SCRIPT))
    parser.add_argument("--train-script", type=str, default=DEFAULT_TRAIN_SCRIPT)
    parser.add_argument("--test-script", type=str, default=DEFAULT_TEST_SCRIPT)
    parser.add_argument("--metadata-dir", type=str, default=DEFAULT_METADATA_DIR)
    parser.add_argument("--robust-run-dirs", nargs="*", default=[])
    parser.add_argument("--robust-run-glob", type=str, default=DEFAULT_ROBUST_RUN_GLOB)
    parser.add_argument("--checkpoint-name", type=str, default="best.pt")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--attack", choices=["pgd", "transfer_pgd", "square", "nes", "hsj", "boundary"], default=None, help="Deprecated single-attack option. If set, overrides --attacks.")
    parser.add_argument("--attacks", nargs="+", choices=["pgd", "transfer_pgd", "square", "nes", "hsj", "boundary"], default=list(DEFAULT_ATTACKS))
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:1" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eps", type=float, default=0.05)
    parser.add_argument("--pgd-steps", type=int, default=50)
    parser.add_argument("--pgd-step-size", type=float, default=0.0)
    parser.add_argument("--pgd-restarts", type=int, default=5)
    parser.add_argument("--queries", type=int, default=2000)
    parser.add_argument("--square-p-init", type=float, default=0.20)
    parser.add_argument("--nes-lr", type=float, default=0.01)
    parser.add_argument("--nes-sigma", type=float, default=0.001)
    parser.add_argument("--nes-samples", type=int, default=20)
    parser.add_argument("--decision-steps", type=int, default=200)
    parser.add_argument("--hsj-initial-gradient-eval-steps", type=int, default=20)
    parser.add_argument("--hsj-max-gradient-eval-steps", type=int, default=200)
    parser.add_argument("--hsj-constraint", choices=["linf", "l2"], default="l2")
    parser.add_argument("--decision-init-eps", type=float, default=0.30)
    parser.add_argument("--decision-init-pgd-steps", type=int, default=100)
    parser.add_argument("--decision-init-pgd-step-size", type=float, default=0.0)
    parser.add_argument("--decision-init-restarts", type=int, default=10)
    parser.add_argument("--sur-epochs", type=int, default=50)
    parser.add_argument("--sur-batch-size", type=int, default=32)
    parser.add_argument("--sur-lr", type=float, default=1e-3)
    args = parser.parse_args()

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir).resolve()
    csv_dir = output_dir / "csv"
    plots_dir = output_dir / "plots"
    ensure_dir(csv_dir)
    ensure_dir(plots_dir)

    robust_module = import_module_from_path("bdd100k_router_robustify_train_eval_module", Path(args.robustify_script).resolve())
    robust_module.ensure_train_import_dependencies()
    train_module = import_module_from_path("bdd100k_moe_train_robustify_eval_module", Path(args.train_script).resolve())
    eval_module = import_module_from_path("bdd100k_moe_test_robustify_eval_module", Path(args.test_script).resolve())
    metadata_dir = Path(args.metadata_dir).resolve()
    bundle = load_json(metadata_dir / "metadata_bundle.json")
    weather_names = [k for k, _ in sorted(bundle["weather_to_id"].items(), key=lambda kv: kv[1])]
    scene_names = [k for k, _ in sorted(bundle["scene_to_id"].items(), key=lambda kv: kv[1])]
    time_names = [k for k, _ in sorted(bundle["time_to_id"].items(), key=lambda kv: kv[1])]
    rows = load_json(metadata_dir / "test_fixed.json")
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    attacks = [args.attack] if args.attack else list(args.attacks)

    summary_rows: List[Dict[str, Any]] = []
    adv_fused_summary_rows: List[Dict[str, Any]] = []
    adv_router_per_class_rows: List[Dict[str, Any]] = []
    adv_router_condition_rows: List[Dict[str, Any]] = []
    coverage_rows: List[Dict[str, Any]] = []
    margin_rows: List[Dict[str, Any]] = []
    prediction_rows: List[Dict[str, Any]] = []
    robust_run_dirs = list_run_dirs(args.robust_run_dirs, args.robust_run_glob)
    print(f"[{timestamp()}] Evaluating {len(robust_run_dirs)} robustified run(s), {len(rows)} image(s), attacks={attacks}", flush=True)
    for robust_run_dir in robust_run_dirs:
        print(f"[{timestamp()}] Robustified run: {robust_run_dir}", flush=True)
        model, thresholds, base_config = load_robust_model(robust_module, train_module, bundle, robust_run_dir, args.checkpoint_name, device)
        image_size = int(base_config.get("image_size", 224))
        pixel_dataset = BDD100KPixelEvalDataset(rows, image_size=image_size)
        clean_loader = make_loader(pixel_dataset, args.batch_size, args.num_workers, device)
        clean_labels = collect_clean_labels(model, clean_loader, device)
        clean_preds = collect_predictions(model, clean_loader, device, thresholds)
        seed = int(base_config.get("seed", args.seed))
        run_actions_dir = output_dir / "actions" / f"{robust_run_dir.name}_seed_{seed}"
        write_csv(
            run_actions_dir / "clean" / "actions.csv",
            build_actions_rows(clean_preds, seed, robust_run_dir, "clean", args.threshold),
        )

        for attack in attacks:
            args.current_attack = attack
            print(f"[{timestamp()}] Robustified run {robust_run_dir.name}: attack={attack}", flush=True)
            adv_images, success, initialized = craft_attack_images(model, clean_loader, clean_labels, args, device, pixel_dataset)
            adv_loader = make_loader(DictTensorDataset(adv_images, pixel_dataset), args.batch_size, args.num_workers, device)
            adv_preds = collect_predictions(model, adv_loader, device, thresholds)
            adv_tables = compute_eval_tables(
                adv_preds,
                eval_module,
                weather_names,
                scene_names,
                time_names,
                args.threshold,
                seed,
                robust_run_dir,
                attack,
                "adv",
            )
            adv_fused_summary_rows.extend(adv_tables["fused_summary"])
            adv_router_per_class_rows.extend(adv_tables["router_per_class"])
            adv_router_condition_rows.extend(adv_tables["router_condition"])
            write_csv(
                run_actions_dir / f"adv_{attack}" / "actions.csv",
                build_actions_rows(adv_preds, seed, robust_run_dir, attack, args.threshold, success=success, initialized=initialized),
            )

            summary_rows.extend(
                compute_after_defense_summary(
                    clean_preds,
                    adv_preds,
                    success,
                    initialized,
                    seed,
                    robust_run_dir,
                    attack,
                    args.threshold,
                    args.topk,
                )
            )
            for row in coverage_vs_robustness_curve(adv_preds, args.threshold):
                row["seed"] = seed
                row["run_dir"] = str(robust_run_dir)
                row["attack"] = attack
                coverage_rows.append(row)
            margin_rows.extend(margin_plot_rows_and_files(clean_preds, adv_preds, plots_dir, seed, robust_run_dir, attack))

            clean_top1 = np.argmax(clean_preds["alpha"], axis=1)
            adv_top1 = np.argmax(adv_preds["alpha"], axis=1)
            router_margin_clean = router_reference_margins(clean_preds["alpha"], clean_top1)
            router_margin_adv = router_reference_margins(adv_preds["alpha"], clean_top1)
            for i, image_path in enumerate(clean_preds["image_path"]):
                prediction_rows.append(
                    {
                        "run": robust_run_dir.name,
                        "run_dir": str(robust_run_dir),
                        "seed": seed,
                        "attack": attack,
                        "image_path": image_path,
                        "router_attack_success": bool(success[i]),
                        "attack_initialized": bool(initialized[i]),
                        "sentinel_action": action_name(int(adv_preds["action_id"][i])),
                        "clean_router_top1": EXPERT_INDEX_TO_NAME[int(clean_top1[i])],
                        "adv_router_top1": EXPERT_INDEX_TO_NAME[int(adv_top1[i])],
                        "clean_risk": float(clean_preds["risk"][i]),
                        "adv_risk": float(adv_preds["risk"][i]),
                        "clean_attack_prob": float(clean_preds["attack_prob"][i]),
                        "adv_attack_prob": float(adv_preds["attack_prob"][i]),
                        "clean_alpha_weather": float(clean_preds["alpha"][i, 0]),
                        "clean_alpha_scene": float(clean_preds["alpha"][i, 1]),
                        "clean_alpha_time": float(clean_preds["alpha"][i, 2]),
                        "adv_alpha_weather": float(adv_preds["alpha"][i, 0]),
                        "adv_alpha_scene": float(adv_preds["alpha"][i, 1]),
                        "adv_alpha_time": float(adv_preds["alpha"][i, 2]),
                        "clean_router_margin": float(router_margin_clean[i]),
                        "adv_router_margin": float(router_margin_adv[i]),
                        "router_margin_degradation": float(router_margin_clean[i] - router_margin_adv[i]),
                        "clean_fused_prob_car": float(clean_preds["probs_fused"][i, 0]),
                        "clean_fused_prob_pedestrian": float(clean_preds["probs_fused"][i, 1]),
                        "clean_fused_prob_traffic_sign": float(clean_preds["probs_fused"][i, 2]),
                        "adv_fused_prob_car": float(adv_preds["probs_fused"][i, 0]),
                        "adv_fused_prob_pedestrian": float(adv_preds["probs_fused"][i, 1]),
                        "adv_fused_prob_traffic_sign": float(adv_preds["probs_fused"][i, 2]),
                        "fused_positive_prob_drop_car": float(clean_preds["probs_fused"][i, 0] - adv_preds["probs_fused"][i, 0]) if clean_preds["y_true"][i, 0] == 1 else float("nan"),
                        "fused_positive_prob_drop_pedestrian": float(clean_preds["probs_fused"][i, 1] - adv_preds["probs_fused"][i, 1]) if clean_preds["y_true"][i, 1] == 1 else float("nan"),
                        "fused_positive_prob_drop_traffic_sign": float(clean_preds["probs_fused"][i, 2] - adv_preds["probs_fused"][i, 2]) if clean_preds["y_true"][i, 2] == 1 else float("nan"),
                        "true_car": int(clean_preds["y_true"][i, 0]),
                        "true_pedestrian": int(clean_preds["y_true"][i, 1]),
                        "true_traffic_sign": int(clean_preds["y_true"][i, 2]),
                    }
                )

    eval_module.write_csv(csv_dir / "robustness_summary_after_defense_per_run.csv", summary_rows)
    eval_module.write_csv(csv_dir / "robustness_summary_after_defense_aggregate.csv", eval_module.summarize_table(summary_rows, key_cols=["attack"]))
    aggregate_and_write(eval_module, csv_dir, "adv_fused_summary", adv_fused_summary_rows, key_cols=["name", "attack", "split"])
    aggregate_and_write(eval_module, csv_dir, "adv_router_per_class", adv_router_per_class_rows, key_cols=["object", "attack", "split"])
    aggregate_and_write(eval_module, csv_dir, "adv_router_condition", adv_router_condition_rows, key_cols=["task", "subset", "attack", "split"])
    eval_module.write_csv(csv_dir / "coverage_vs_robustness_per_run.csv", coverage_rows)
    eval_module.write_csv(csv_dir / "coverage_vs_robustness_aggregate.csv", eval_module.summarize_table(coverage_rows, key_cols=["attack", "risk_threshold"]))
    eval_module.write_csv(csv_dir / "margin_and_drop_summary_per_run.csv", margin_rows)
    eval_module.write_csv(csv_dir / "margin_and_drop_summary_aggregate.csv", eval_module.summarize_table(margin_rows, key_cols=["attack", "name"]))
    eval_module.write_csv(csv_dir / f"predictions_robustified_seed_{args.seed}.csv", prediction_rows)

    manifest = vars(args).copy()
    manifest.pop("current_attack", None)
    manifest["attacks_evaluated"] = attacks
    manifest["router_label_definition"] = "post_robustification_clean_router_top1"
    manifest["metrics_methodology"] = "post_defense_router_sentinel_summary_with_adversarial_eval_router_and_fused_tables"
    save_json(manifest, output_dir / "robustify_eval_manifest.json")
    print(f"[{timestamp()}] Robustified router evaluation complete. CSV files saved to: {csv_dir}", flush=True)
    print(f"[{timestamp()}] Margin/drop plots saved to: {plots_dir}", flush=True)


if __name__ == "__main__":
    main()
