import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor, nn, optim
from torch.amp import GradScaler
from torch.optim.lr_scheduler import MultiStepLR
from torchvision.io import decode_image
from torchvision.transforms import v2 as transforms

from config import create_logger


@dataclass
class Metrics:
    epochs: int = field(default=0)
    generator_learning_rates: list[float] = field(default_factory=list)
    discriminator_learning_rates: list[float] = field(default_factory=list)
    generator_train_losses: list[float] = field(default_factory=list)
    discriminator_train_losses: list[float] = field(default_factory=list)
    generator_val_losses: list[float] = field(default_factory=list)
    generator_val_psnrs: list[float] = field(default_factory=list)
    generator_val_ssims: list[float] = field(default_factory=list)


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


def _save_optimizer_state(
    optimizer: optim.Optimizer,
    checkpoint_dir_path: str | Path,
    prefix: str,
) -> dict:
    checkpoint_dir_path = Path(checkpoint_dir_path)

    optimizer_state = optimizer.state_dict()
    optimizer_tensors = {}
    optimizer_metadata = {"param_groups": optimizer_state["param_groups"]}

    optimizer_state_buffers = optimizer_state["state"]
    optimizer_metadata["state"] = {}

    for param_id, buffers in optimizer_state_buffers.items():
        param_id_str = str(param_id)
        optimizer_metadata["state"][param_id_str] = {}

        for buffer_name, value in buffers.items():
            if isinstance(value, torch.Tensor):
                tensor_key = f"state_{param_id_str}_{buffer_name}"
                optimizer_tensors[tensor_key] = value
            else:
                optimizer_metadata["state"][param_id_str][buffer_name] = value

    if optimizer_tensors:
        save_file(
            optimizer_tensors, checkpoint_dir_path / f"{prefix}_optimizer.safetensors"
        )

    return optimizer_metadata


def _load_optimizer_state(
    optimizer: optim.Optimizer,
    checkpoint_dir_path: str | Path,
    prefix: str,
    full_metadata: dict,
    device: str,
) -> None:
    checkpoint_dir_path = Path(checkpoint_dir_path)

    optimizer_metadata_key = f"{prefix}_optimizer_metadata"
    if optimizer_metadata_key not in full_metadata:
        logger.warning(
            f"Metadata for '{optimizer_metadata_key}' not found in training_state.json"
        )
        return

    optimizer_metadata = full_metadata[optimizer_metadata_key]

    optimizer_tensors_path = checkpoint_dir_path / f"{prefix}_optimizer.safetensors"
    if optimizer_tensors_path.exists():
        optimizer_tensors = load_file(filename=optimizer_tensors_path, device=device)
    else:
        optimizer_tensors = {}
        logger.warning(f"Optimizer tensor file not found: {optimizer_tensors_path}")

    optimizer_state_buffers = {
        int(param_id): buffers
        for param_id, buffers in optimizer_metadata["state"].items()
    }

    for tensor_key, tensor_value in optimizer_tensors.items():
        parts = tensor_key.split("_")
        if len(parts) < 3 or parts[0] != "state":
            logger.warning(
                f"Unrecognized tensor key in {prefix}_optimizer: {tensor_key}"
            )
            continue

        param_id = int(parts[1])
        buffer_name = "_".join(parts[2:])

        if param_id not in optimizer_state_buffers:
            optimizer_state_buffers[param_id] = {}

        optimizer_state_buffers[param_id][buffer_name] = tensor_value

    optimizer_state_to_load = {
        "param_groups": optimizer_metadata["param_groups"],
        "state": optimizer_state_buffers,
    }

    try:
        optimizer.load_state_dict(optimizer_state_to_load)
    except Exception as e:
        logger.error(f"Failed to load state_dict for {prefix}_optimizer: {e}")
        logger.warning(f"Continuing without loading {prefix}_optimizer state.")


def save_checkpoint(
    checkpoint_dir_path: str | Path,
    epoch: int,
    generator: nn.Module,
    discriminator: nn.Module,
    generator_optimizer: optim.Optimizer,
    discriminator_optimizer: optim.Optimizer,
    metrics: Metrics,
    generator_scaler: GradScaler | None = None,
    discriminator_scaler: GradScaler | None = None,
    generator_scheduler: MultiStepLR | None = None,
    discriminator_scheduler: MultiStepLR | None = None,
) -> None:
    checkpoint_dir_path = Path(checkpoint_dir_path)
    checkpoint_dir_path.mkdir(parents=True, exist_ok=True)

    save_file(generator.state_dict(), checkpoint_dir_path / "generator.safetensors")
    save_file(
        discriminator.state_dict(), checkpoint_dir_path / "discriminator.safetensors"
    )

    generator_optimizer_metadata = _save_optimizer_state(
        generator_optimizer,
        checkpoint_dir_path,
        "generator",
    )

    discriminator_optimizer_metadata = _save_optimizer_state(
        discriminator_optimizer,
        checkpoint_dir_path,
        "discriminator",
    )

    full_metadata = {
        "epoch": epoch,
        "metrics": asdict(metrics),
        "generator_optimizer_metadata": generator_optimizer_metadata,
        "discriminator_optimizer_metadata": discriminator_optimizer_metadata,
        "generator_scaler_state_dict": generator_scaler.state_dict()
        if generator_scaler
        else None,
        "discriminator_scaler_state_dict": discriminator_scaler.state_dict()
        if discriminator_scaler
        else None,
        "generator_scheduler_state_dict": generator_scheduler.state_dict()
        if generator_scheduler
        else None,
        "discriminator_scheduler_state_dict": discriminator_scheduler.state_dict()
        if discriminator_scheduler
        else None,
    }

    with open(checkpoint_dir_path / "training_state.json", "w") as f:
        json.dump(full_metadata, f, indent=4)

    logger.debug(f'Checkpoint was saved to "{checkpoint_dir_path}" after {epoch} epoch')


def load_checkpoint(
    checkpoint_dir_path: str | Path,
    generator: nn.Module,
    discriminator: nn.Module,
    generator_optimizer: optim.Optimizer,
    discriminator_optimizer: optim.Optimizer,
    metrics: Metrics,
    generator_scaler: GradScaler | None = None,
    discriminator_scaler: GradScaler | None = None,
    generator_scheduler: MultiStepLR | None = None,
    discriminator_scheduler: MultiStepLR | None = None,
    device: Literal["cuda", "cpu"] = "cpu",
) -> int:
    checkpoint_dir_path = Path(checkpoint_dir_path)

    generator_path = checkpoint_dir_path / "generator.safetensors"
    discriminator_path = checkpoint_dir_path / "discriminator.safetensors"
    state_path = checkpoint_dir_path / "training_state.json"

    if (
        not generator_path.exists()
        or not discriminator_path.exists()
        or not state_path.exists()
    ):
        logger.warning(
            f'Checkpoint was not found at "{checkpoint_dir_path}", starting from 1 epoch'
        )
        return 1

    generator.load_state_dict(load_file(filename=generator_path, device=device))
    discriminator.load_state_dict(load_file(filename=discriminator_path, device=device))

    with open(state_path, "r") as f:
        full_metadata = json.load(f)

    if metrics and "metrics" in full_metadata:
        metrics_dict = full_metadata["metrics"]
        metrics.epochs = metrics_dict["epochs"]
        metrics.generator_learning_rates = metrics_dict["generator_learning_rates"]
        metrics.discriminator_learning_rates = metrics_dict[
            "discriminator_learning_rates"
        ]
        metrics.generator_train_losses = metrics_dict["generator_train_losses"]
        metrics.discriminator_train_losses = metrics_dict["discriminator_train_losses"]
        metrics.generator_val_losses = metrics_dict["generator_val_losses"]
        metrics.generator_val_psnrs = metrics_dict["generator_val_psnrs"]
        metrics.generator_val_ssims = metrics_dict["generator_val_ssims"]

    if generator_scaler and full_metadata["generator_scaler_state_dict"]:
        generator_scaler.load_state_dict(full_metadata["generator_scaler_state_dict"])

    if discriminator_scaler and full_metadata["discriminator_scaler_state_dict"]:
        discriminator_scaler.load_state_dict(
            full_metadata["discriminator_scaler_state_dict"]
        )

    if generator_scheduler and full_metadata["generator_scheduler_state_dict"]:
        generator_scheduler.load_state_dict(
            full_metadata["generator_scheduler_state_dict"]
        )

    if discriminator_scheduler and full_metadata["discriminator_scheduler_state_dict"]:
        discriminator_scheduler.load_state_dict(
            full_metadata["discriminator_scheduler_state_dict"]
        )

    _load_optimizer_state(
        generator_optimizer,
        checkpoint_dir_path,
        "generator",
        full_metadata,
        device,
    )

    _load_optimizer_state(
        discriminator_optimizer,
        checkpoint_dir_path,
        "discriminator",
        full_metadata,
        device,
    )

    logger.info(f'Checkpoint was loaded from "{checkpoint_dir_path}"')

    return full_metadata["epoch"]


def format_time(total_seconds: float) -> str:
    if total_seconds < 0:
        total_seconds = 0

    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)

    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def rgb_to_ycbcr(image_tensor: torch.Tensor) -> torch.Tensor:
    if image_tensor.dim() == 4:
        image_tensor.squeeze_(0)

    image_tensor = (image_tensor + 1) / 2

    weights = torch.tensor(
        [0.299, 0.587, 0.114],
        dtype=image_tensor.dtype,
        device=image_tensor.device,
    )

    Y_channel = torch.sum(
        image_tensor * weights.view(1, 3, 1, 1),
        dim=1,
        keepdim=True,
    )

    return Y_channel
