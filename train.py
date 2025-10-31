from pathlib import Path
from time import time
from typing import Literal

import torch
from safetensors.torch import load_file
from torch import nn, optim
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from config import create_logger
from data_processing import SRDataset
from models import Discriminator, Generator, TruncatedVGG19
from utils import Metrics, format_time, load_checkpoint, rgb_to_ycbcr, save_checkpoint

SCALING_FACTOR: Literal[2, 4, 8] = 4
CROP_SIZE = 128

GENERATOR_N_CHANNELS = 96
GENERATOR_N_RES_BLOCKS = 16
LARGE_KERNEL_SIZE = 9
SMALL_KERNEL_SIZE = 3

DISCRIMINATOR_N_CHANNELS = 64
DISCRIMINATOR_KERNEL_SIZE = 3
DISCRIMINATOR_N_CONV_BLOCKS = 8
DISCRIMINATOR_FLL_SIZE = 1024

BATCH_SIZE = 32
LEARNING_RATE = 1e-5
MAX_LEARNING_RATE = 1e-4
EPOCHS = 500
PRINT_FREQ = 200

INITIALIZE_WITH_SRRESNET_CHECKPOINT = True
LOAD_CHECKPOINT = False
LOAD_BEST_CHECKPOINT = False
DEV_MOVE = True

NUM_WORKERS = 8

SCHEDULER_MILESTONES = [EPOCHS // 2]
SCHEDULER_GAMMA = 0.5
BETA = 1e-3

TRAIN_DATASET_PATH = Path("data/DF2K_OST.txt")
VAL_DATASET_PATH = Path("data/DIV2K_valid.txt")

SRRESNET_MODEL_CHECKPOINT_PATH = Path(
    "checkpoints/srresnet_best/srresnet_model_best.safetensors"
)
BEST_CHECKPOINT_DIR_PATH = Path("checkpoints/srgan_best")
CHECKPOINT_DIR_PATH = Path("checkpoints/srgan_latest")

logger = create_logger("INFO", __file__)


def train_step(
    data_loader: DataLoader,
    generator: nn.Module,
    discriminator: nn.Module,
    truncated_vgg19: nn.Module,
    content_loss_fn: nn.Module,
    adversarial_loss_fn: nn.Module,
    generator_optimizer: optim.Optimizer,
    discriminator_optimizer: optim.Optimizer,
    generator_scaler: GradScaler | None = None,
    discriminator_scaler: GradScaler | None = None,
    generator_scheduler: MultiStepLR | None = None,
    discriminator_scheduler: MultiStepLR | None = None,
    device: Literal["cuda", "cpu"] = "cpu",
) -> tuple[float, float]:
    total_generator_loss = 0.0
    total_discrimination_loss = 0.0

    generator.train()
    discriminator.train()

    for i, (hr_img_tensor, lr_img_tensor) in enumerate(data_loader):
        hr_img_tensor = hr_img_tensor.to(device, non_blocking=True)
        lr_img_tensor = lr_img_tensor.to(device, non_blocking=True)

        with autocast(device, enabled=(generator_scaler is not None)):
            sr_img_tensor = generator(lr_img_tensor)

            sr_img_tensor_in_vgg_space = truncated_vgg19(sr_img_tensor)
            hr_img_tensor_in_vgg_space = truncated_vgg19(hr_img_tensor).detach()

            sr_discriminated = discriminator(sr_img_tensor)

            content_loss = content_loss_fn(
                sr_img_tensor_in_vgg_space, hr_img_tensor_in_vgg_space
            )
            adversarial_loss = adversarial_loss_fn(
                sr_discriminated, torch.ones_like(sr_discriminated)
            )
            perceptual_loss = content_loss + BETA * adversarial_loss

        total_generator_loss += perceptual_loss.item()

        generator_optimizer.zero_grad()

        if generator_scaler:
            generator_scaler.scale(perceptual_loss).backward()
            generator_scaler.step(generator_optimizer)
            generator_scaler.update()
        else:
            perceptual_loss.backward()
            generator_optimizer.step()

        with autocast(device, enabled=(discriminator_scaler is not None)):
            hr_discriminated = discriminator(hr_img_tensor)
            sr_discriminated = discriminator(sr_img_tensor.detach())

            adversarial_loss = adversarial_loss_fn(
                sr_discriminated, torch.zeros_like(sr_discriminated)
            ) + adversarial_loss_fn(hr_discriminated, torch.ones_like(hr_discriminated))

        total_discrimination_loss += adversarial_loss.item()

        discriminator_optimizer.zero_grad()

        if discriminator_scaler:
            discriminator_scaler.scale(adversarial_loss).backward()
            discriminator_scaler.step(discriminator_optimizer)
            discriminator_scaler.update()
        else:
            adversarial_loss.backward()
            discriminator_optimizer.step()

        if i % PRINT_FREQ == 0:
            logger.debug(f"Processing batch {i}/{len(data_loader)}...")

    total_generator_loss /= len(data_loader)
    total_discrimination_loss /= len(data_loader)

    return total_generator_loss, total_discrimination_loss


def validation_step(
    data_loader: DataLoader,
    generator: nn.Module,
    truncated_vgg19: nn.Module,
    content_loss_fn: nn.Module,
    psnr_metric: PeakSignalNoiseRatio,
    ssim_metric: StructuralSimilarityIndexMeasure,
    device: Literal["cpu", "cuda"] = "cpu",
) -> tuple[float, float, float]:
    total_content_loss = 0.0
    total_psnr = 0
    total_ssim = 0

    generator.eval()

    with torch.inference_mode():
        for hr_img_tensor, lr_img_tensor in data_loader:
            hr_img_tensor = hr_img_tensor.to(device, non_blocking=True)
            lr_img_tensor = lr_img_tensor.to(device, non_blocking=True)

            sr_img_tensor = generator(lr_img_tensor)

            sr_img_tensor_in_vgg_space = truncated_vgg19(sr_img_tensor)
            hr_img_tensor_in_vgg_space = truncated_vgg19(hr_img_tensor)

            content_loss = content_loss_fn(
                sr_img_tensor_in_vgg_space, hr_img_tensor_in_vgg_space
            )

            y_hr_tensor = rgb_to_ycbcr(hr_img_tensor)
            y_sr_tensor = rgb_to_ycbcr(sr_img_tensor)

            sf = SCALING_FACTOR
            y_hr_tensor = y_hr_tensor[:, :, sf:-sf, sf:-sf]
            y_sr_tensor = y_sr_tensor[:, :, sf:-sf, sf:-sf]

            psnr = psnr_metric(y_sr_tensor, y_hr_tensor)
            ssim = ssim_metric(y_sr_tensor, y_hr_tensor)

            total_content_loss += content_loss.item()
            total_psnr += psnr.item()
            total_ssim += ssim.item()

        total_content_loss /= len(data_loader)
        total_psnr /= len(data_loader)
        total_ssim /= len(data_loader)

    return total_content_loss, total_psnr, total_ssim


def train(
    train_data_loader: DataLoader,
    val_data_loader: DataLoader,
    generator: nn.Module,
    discriminator: nn.Module,
    truncated_vgg19: nn.Module,
    content_loss_fn: nn.Module,
    adversarial_loss_fn: nn.Module,
    generator_optimizer: optim.Optimizer,
    discriminator_optimizer: optim.Optimizer,
    start_epoch: int,
    epochs: int,
    metrics: Metrics,
    psnr_metric: PeakSignalNoiseRatio,
    ssim_metric: StructuralSimilarityIndexMeasure,
    generator_scaler: GradScaler | None = None,
    discriminator_scaler: GradScaler | None = None,
    generator_scheduler: MultiStepLR | None = None,
    discriminator_scheduler: MultiStepLR | None = None,
    device: Literal["cuda", "cpu"] = "cpu",
) -> None:
    if not metrics.epochs:
        metrics.epochs = epochs - start_epoch + 1

    if start_epoch > 1 and metrics.generator_val_losses:
        best_val_loss = min(metrics.generator_val_losses)
    else:
        best_val_loss = float("inf")

    logger.info("-" * 107)
    logger.info("Model parameters:")
    logger.info(f"Scaling factor: {SCALING_FACTOR}")
    logger.info("-" * 107)
    logger.info("Starting model training...")

    try:
        for epoch in range(start_epoch, epochs + 1):
            start_time = time()

            generator_train_loss, discriminator_train_loss = train_step(
                data_loader=train_data_loader,
                generator=generator,
                discriminator=discriminator,
                truncated_vgg19=truncated_vgg19,
                content_loss_fn=content_loss_fn,
                adversarial_loss_fn=adversarial_loss_fn,
                generator_optimizer=generator_optimizer,
                discriminator_optimizer=discriminator_optimizer,
                generator_scaler=generator_scaler,
                discriminator_scaler=discriminator_scaler,
                generator_scheduler=generator_scheduler,
                discriminator_scheduler=discriminator_scheduler,
                device=device,
            )

            generator_val_loss, generator_val_psnr, generator_val_ssim = (
                validation_step(
                    data_loader=val_data_loader,
                    generator=generator,
                    truncated_vgg19=truncated_vgg19,
                    content_loss_fn=content_loss_fn,
                    psnr_metric=psnr_metric,
                    ssim_metric=ssim_metric,
                    device=device,
                )
            )

            if generator_scheduler:
                generator_scheduler.step()

            if discriminator_scheduler:
                discriminator_scheduler.step()

            end_time = time() - start_time
            epoch_time = format_time(end_time)
            remaining_time = format_time(end_time * (epochs - epoch))

            generator_optimizer_lr = generator_optimizer.param_groups[0]["lr"]
            discriminator_optimizer_lr = discriminator_optimizer.param_groups[0]["lr"]

            metrics.generator_learning_rates.append(generator_optimizer_lr)
            metrics.discriminator_learning_rates.append(discriminator_optimizer_lr)
            metrics.generator_train_losses.append(generator_train_loss)
            metrics.discriminator_train_losses.append(discriminator_train_loss)
            metrics.generator_val_losses.append(generator_val_loss)
            metrics.generator_val_psnrs.append(generator_val_psnr)
            metrics.generator_val_ssims.append(generator_val_ssim)

            logger.info(
                f"Epoch: {epoch}/{epochs} ({epoch_time}/{remaining_time}) | Generator LR: {generator_optimizer_lr} | Discriminator LR: {discriminator_optimizer_lr}"
            )
            logger.info(
                f"Generator Train Loss: {generator_train_loss:.4f} | Discriminator Train Loss: {discriminator_train_loss:.4f} | Generator Val Loss: {generator_val_loss:.4f} | Generator Val PSNR: {generator_val_psnr:.4f} | Generator Val SSIM: {generator_val_ssim:.4f}"
            )

            if generator_val_loss < best_val_loss:
                best_val_loss = generator_val_loss
                logger.debug(
                    f"New best model found with val loss: {best_val_loss:.4f} at epoch {epoch}"
                )
                save_checkpoint(
                    checkpoint_dir_path=BEST_CHECKPOINT_DIR_PATH,
                    epoch=epoch,
                    generator=generator,
                    discriminator=discriminator,
                    generator_optimizer=generator_optimizer,
                    discriminator_optimizer=discriminator_optimizer,
                    metrics=metrics,
                    generator_scaler=generator_scaler,
                    discriminator_scaler=discriminator_scaler,
                    generator_scheduler=generator_scheduler,
                    discriminator_scheduler=discriminator_scheduler,
                )

            save_checkpoint(
                checkpoint_dir_path=CHECKPOINT_DIR_PATH,
                epoch=epoch,
                generator=generator,
                discriminator=discriminator,
                generator_optimizer=generator_optimizer,
                discriminator_optimizer=discriminator_optimizer,
                metrics=metrics,
                generator_scaler=generator_scaler,
                discriminator_scaler=discriminator_scaler,
                generator_scheduler=generator_scheduler,
                discriminator_scheduler=discriminator_scheduler,
            )

    except KeyboardInterrupt:
        logger.info("Saving model's weights and finish training...")
        save_checkpoint(
            checkpoint_dir_path=CHECKPOINT_DIR_PATH,
            epoch=epoch,
            generator=generator,
            discriminator=discriminator,
            generator_optimizer=generator_optimizer,
            discriminator_optimizer=discriminator_optimizer,
            metrics=metrics,
            generator_scaler=generator_scaler,
            discriminator_scaler=discriminator_scaler,
            generator_scheduler=generator_scheduler,
            discriminator_scheduler=discriminator_scheduler,
        )


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = SRDataset(
        data_path=TRAIN_DATASET_PATH,
        scaling_factor=SCALING_FACTOR,
        crop_size=CROP_SIZE,
        dev_mode=DEV_MOVE,
    )

    val_dataset = SRDataset(
        data_path=VAL_DATASET_PATH,
        scaling_factor=SCALING_FACTOR,
        crop_size=CROP_SIZE,
        dev_mode=DEV_MOVE,
    )

    train_data_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True if device == "cuda" else False,
        num_workers=NUM_WORKERS,
    )

    val_data_loader = DataLoader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True if device == "cuda" else False,
        num_workers=NUM_WORKERS,
    )

    generator = Generator(
        n_channels=GENERATOR_N_CHANNELS,
        large_kernel_size=LARGE_KERNEL_SIZE,
        small_kernel_size=SMALL_KERNEL_SIZE,
        n_res_blocks=GENERATOR_N_RES_BLOCKS,
        scaling_factor=SCALING_FACTOR,
    ).to(device)

    discriminator = Discriminator(
        n_channels=DISCRIMINATOR_N_CHANNELS,
        kernel_size=DISCRIMINATOR_KERNEL_SIZE,
        n_conv_blocks=DISCRIMINATOR_N_CONV_BLOCKS,
        first_linear_layer_size=DISCRIMINATOR_FLL_SIZE,
    ).to(device)

    truncated_vgg19 = TruncatedVGG19().to(device)

    content_loss_fn = nn.MSELoss()
    adversarial_loss_fn = nn.BCEWithLogitsLoss()

    metrics = Metrics()
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    generator_optimizer = optim.Adam(generator.parameters(), lr=LEARNING_RATE)
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE)

    generator_scaler = GradScaler(device) if device == "cuda" else None
    discriminator_scaler = GradScaler(device) if device == "cuda" else None

    generator_scheduler = MultiStepLR(
        optimizer=generator_optimizer,
        milestones=SCHEDULER_MILESTONES,
        gamma=SCHEDULER_GAMMA,
    )
    discriminator_scheduler = MultiStepLR(
        optimizer=discriminator_optimizer,
        milestones=SCHEDULER_MILESTONES,
        gamma=SCHEDULER_GAMMA,
    )

    if INITIALIZE_WITH_SRRESNET_CHECKPOINT and SRRESNET_MODEL_CHECKPOINT_PATH.exists():
        generator_weights = load_file(
            filename=SRRESNET_MODEL_CHECKPOINT_PATH, device=device
        )
        generator.load_state_dict(generator_weights)
        logger.info("Successfully loaded pre-trained SRResNet weights into Generator")

    start_epoch = 1
    if LOAD_CHECKPOINT:
        if BEST_CHECKPOINT_DIR_PATH.exists() or CHECKPOINT_DIR_PATH.exists():
            if LOAD_BEST_CHECKPOINT and BEST_CHECKPOINT_DIR_PATH.exists():
                checkpoint_dir_path_to_load = BEST_CHECKPOINT_DIR_PATH
                logger.debug(
                    f'Loading best checkpoint from "{checkpoint_dir_path_to_load}"...'
                )
            elif CHECKPOINT_DIR_PATH.exists():
                checkpoint_dir_path_to_load = CHECKPOINT_DIR_PATH
                logger.debug(
                    f'Loading checkpoint from "{checkpoint_dir_path_to_load}"...'
                )

            start_epoch = load_checkpoint(
                checkpoint_dir_path=checkpoint_dir_path_to_load,
                generator=generator,
                discriminator=discriminator,
                generator_optimizer=generator_optimizer,
                discriminator_optimizer=discriminator_optimizer,
                metrics=metrics,
                generator_scaler=generator_scaler,
                discriminator_scaler=discriminator_scaler,
                generator_scheduler=generator_scheduler,
                discriminator_scheduler=discriminator_scheduler,
                device=device,
            )

            if generator_scheduler and discriminator_scheduler and start_epoch > 1:
                epochs_to_skip = start_epoch - 1

                for _ in range(epochs_to_skip):
                    generator_scheduler.step()
                    discriminator_scheduler.step()

                logger.info(f"Schedulers advanced to epoch {start_epoch}")
        else:
            logger.info(
                "No checkpoints were found, start training from the beginning..."
            )

    train(
        train_data_loader=train_data_loader,
        val_data_loader=val_data_loader,
        generator=generator,
        discriminator=discriminator,
        truncated_vgg19=truncated_vgg19,
        content_loss_fn=content_loss_fn,
        adversarial_loss_fn=adversarial_loss_fn,
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        start_epoch=start_epoch,
        epochs=EPOCHS,
        metrics=metrics,
        psnr_metric=psnr_metric,
        ssim_metric=ssim_metric,
        generator_scaler=generator_scaler,
        discriminator_scaler=discriminator_scaler,
        generator_scheduler=generator_scheduler,
        discriminator_scheduler=discriminator_scheduler,
        device=device,
    )


if __name__ == "__main__":
    main()
