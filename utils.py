import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor, nn, optim
from torch.amp import GradScaler
from torch.optim.lr_scheduler import OneCycleLR
from torchvision.io import decode_image
from torchvision.transforms import v2 as transforms

from config import create_logger


@dataclass
class Metrics:
    epochs: int = field(default=0)
    learning_rates: list[float] = field(default_factory=list)
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    psnrs: list[float] = field(default_factory=list)
    ssims: list[float] = field(default_factory=list)


logger = create_logger("INFO", __file__)


def create_hr_and_lr_imgs(
    img_path: str | Path,
    scaling_factor: Literal[2, 4, 8],
    crop_size: int | None = None,
    test_mode: bool = False,
) -> tuple[Tensor, Tensor]:
    img_tensor = decode_image(Path(img_path).__fspath__())

    if test_mode:
        _, height, width = img_tensor.shape

        height_remainder = height % scaling_factor
        width_remainder = width % scaling_factor

        top_bound = height_remainder // 2
        left_bound = width_remainder // 2

        bottom_bound = top_bound + (height - height_remainder)
        right_bound = left_bound + (width - width_remainder)

        hr_img_tensor = img_tensor[:, top_bound:bottom_bound, left_bound:right_bound]
    elif crop_size:
        augmentation_transforms = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomChoice(
                    [
                        transforms.RandomRotation(degrees=[0, 0]),
                        transforms.RandomRotation(degrees=[90, 90]),
                        transforms.RandomRotation(degrees=[180, 180]),
                        transforms.RandomRotation(degrees=[270, 270]),
                    ]
                ),
                transforms.RandomCrop(size=(crop_size, crop_size)),
            ]
        )

        hr_img_tensor = augmentation_transforms(img_tensor)

    lr_transforms = transforms.Compose(
        [
            transforms.Resize(
                size=(
                    hr_img_tensor.shape[1] // scaling_factor,
                    hr_img_tensor.shape[2] // scaling_factor,
                ),
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True,
            )
        ]
    )

    normalize_transforms = transforms.Compose(
        [
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    lr_img_tensor = lr_transforms(hr_img_tensor)

    hr_img_tensor = normalize_transforms(hr_img_tensor)
    lr_img_tensor = normalize_transforms(lr_img_tensor)

    return hr_img_tensor, lr_img_tensor


def save_checkpoint(
    checkpoints_dir_path: str | Path,
    epoch: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    metrics: Metrics,
    scaler: GradScaler | None = None,
    scheduler: OneCycleLR | None = None,
) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    checkpoint_dir_path = Path(checkpoints_dir_path) / f"checkpoint_{timestamp}"
    checkpoint_dir_path.mkdir(parents=True, exist_ok=True)

    save_file(model.state_dict(), checkpoint_dir_path / "model.safetensors")

    optimizer_state = optimizer.state_dict()
    optimizer_tensors = {}
    optimizer_metadata = {}

    optimizer_metadata["param_groups"] = optimizer_state["param_groups"]

    optimizer_state_buffers = optimizer_state["state"]
    optimizer_metadata["state"] = {}

    for param_id, buffers in optimizer_state_buffers.items():
        param_id = str(param_id)
        optimizer_metadata["state"][param_id] = {}

        for buffer_name, value in buffers.items():
            if isinstance(value, torch.Tensor):
                tensor_key = f"state_{param_id}_{buffer_name}"
                optimizer_tensors[tensor_key] = value
            else:
                optimizer_metadata["state"][param_id][buffer_name] = value

    if optimizer_tensors:
        save_file(optimizer_tensors, checkpoint_dir_path / "optimizer.safetensors")

    full_metadata = {
        "epoch": epoch,
        "metrics": asdict(metrics),
        "optimizer_metadata": optimizer_metadata,
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
    }

    with open(checkpoint_dir_path / "training_state.json", "w") as f:
        json.dump(full_metadata, f, indent=4)

    logger.info(f'Checkpoint was saved at "{checkpoint_dir_path}" after {epoch} epoch')


def load_checkpoint(
    checkpoints_dir_path: str | Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    metrics: Metrics | None = None,
    scaler: GradScaler | None = None,
    scheduler: OneCycleLR | None = None,
    device: Literal["cuda", "cpu"] = "cpu",
) -> int:
    checkpoints_dir_path = Path(checkpoints_dir_path)

    model_path = checkpoints_dir_path / "model.safetensors"
    state_path = checkpoints_dir_path / "training_state.json"

    if not model_path.exists() or not state_path.exists():
        logger.info(
            'Checkpoint was not found at "{checkpoints_dir_path}", starting from 1 epoch'
        )
        return 1

    model.load_state_dict(load_file(filename=model_path, device=device))

    with open("state_path", "r") as f:
        full_metadata = json.load(f)

    if metrics and "metrics" in full_metadata:
        metrics_dict = full_metadata["metrics"]
        metrics.epochs = metrics_dict["epochs"]
        metrics.learning_rates = metrics_dict["learning_rates"]
        metrics.train_losses = metrics_dict["train_losses"]
        metrics.val_losses = metrics_dict["val_losses"]
        metrics.psnrs = metrics_dict["psnrs"]
        metrics.ssims = metrics_dict["ssims"]

    if scaler and full_metadata["scaler_state_dict"]:
        scaler.load_state_dict(full_metadata["scaler_state_dict"])

    if scheduler and full_metadata["scheduler_state_dict"]:
        scheduler.load_state_dict(full_metadata["scheduler_state_dict"])

    optimizer_tensors_path = checkpoints_dir_path / "optimizer.safetensors"

    if optimizer_tensors_path.exists():
        optimizer_tensors = load_file(filename=optimizer_tensors_path, device=device)
    else:
        optimizer_tensors = {}

    optimizer_metadata = full_metadata["optimizer_metadata"]

    optimizer_state_buffers = {
        int(param_id): buffers
        for param_id, buffers in optimizer_metadata["state"].items()
    }

    for param_id, buffers in optimizer_state_buffers.items():
        param_id = str(param_id)

        for buffer_name in list(buffers.keys()):
            tensor_key = f"state_{param_id}_{buffer_name}"
            if tensor_key in optimizer_tensors:
                buffers[buffer_name] = optimizer_tensors[tensor_key]

        for tensor_key, tensor_value in optimizer_tensors.items():
            if tensor_key.startswith(f"state_{param_id}_"):
                original_buffer_name = tensor_key[len(f"state_{param_id}_") :]
                if original_buffer_name not in buffers:
                    buffers[original_buffer_name] = tensor_value

    optimizer_state_to_load = {
        "param_groups": optimizer_metadata["param_groups"],
        "state": optimizer_state_buffers,
    }

    optimizer.load_state_dict(optimizer_state_to_load)

    logger.info('Checkpoint was loaded from "{checkpoints_dir_path}"')

    return full_metadata["epoch"]
