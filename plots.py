from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from utils import Metrics


def plot_training_metrics(metrics: Metrics, hyperparameters_str: str) -> None:
    sns.set_style("whitegrid")
    sns.set_palette("deep")
    palette = sns.color_palette("deep")

    epochs = list(range(0, len(metrics.generator_train_losses)))

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("SRGAN Training Metrics", fontsize=18)

    fig.text(0.5, 0.94, hyperparameters_str, ha="center", va="top", fontsize=10)

    sns.lineplot(
        x=epochs,
        y=metrics.generator_train_losses,
        label="Generator train loss",
        ax=axs[0, 0],
        linewidth=2.5,
        color=palette[0],
    )
    sns.lineplot(
        x=epochs,
        y=metrics.generator_val_losses,
        label="Generator val loss",
        ax=axs[0, 0],
        linewidth=2.5,
        color=palette[1],
    )
    axs[0, 0].set_title("Generator training and validation losses")
    axs[0, 0].set_xlabel("Epoch")
    axs[0, 0].set_ylabel("Loss")

    sns.lineplot(
        x=epochs,
        y=metrics.discriminator_train_losses,
        ax=axs[0, 1],
        linewidth=2.5,
        color=palette[2],
    )
    axs[0, 1].set_title("Discriminator training loss")
    axs[0, 1].set_xlabel("Epoch")
    axs[0, 1].set_ylabel("Loss")

    sns.lineplot(
        x=epochs,
        y=metrics.generator_val_psnrs,
        ax=axs[1, 0],
        linewidth=2.5,
        color=palette[1],
    )
    axs[1, 0].set_title("Validation Peak Signal-to-Noise Ratio")
    axs[1, 0].set_xlabel("Epoch")
    axs[1, 0].set_ylabel("PSNR")

    sns.lineplot(
        x=epochs,
        y=metrics.generator_val_ssims,
        ax=axs[1, 1],
        linewidth=2.5,
        color=palette[1],
    )
    axs[1, 1].set_title("Validation Structural Similarity Index Measure")
    axs[1, 1].set_xlabel("Epoch")
    axs[1, 1].set_ylabel("SSIM")

    plt.tight_layout(rect=[0, 0.03, 1, 0.94])

    output_path = (
        Path("images")
        / f"training_metrics_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.show()
