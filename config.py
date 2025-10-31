import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime
from typing import Literal

SCALING_FACTOR: Literal[2, 4, 8] = 4
CROP_SIZE = 128

GENERATOR_CHANNELS_COUNT = 96
GENERATOR_RES_BLOCKS_COUNT = 16
GENERATOR_LARGE_KERNEL_SIZE = 9
GENERATOR_SMALL_KERNEL_SIZE = 3

DISCRIMINATOR_CHANNELS_COUNT = 64
DISCRIMINATOR_KERNEL_SIZE = 3
DISCRIMINATOR_CONV_BLOCKS_COUNT = 8
DISCRIMINATOR_LINEAR_LAYER_SIZE = 1024

TRAIN_BATCH_SIZE = 32
TEST_BATCH_SIZE = 1
LEARNING_RATE = 1e-5
MAX_LEARNING_RATE = 1e-4
EPOCHS = 500
PRINT_FREQUENCY = 200

INITIALIZE_WITH_SRRESNET_CHECKPOINT = True
LOAD_CHECKPOINT = False
LOAD_BEST_CHECKPOINT = False
DEV_MOVE = True

SCHEDULER_MILESTONES = [EPOCHS // 2]
SCHEDULER_GAMMA = 0.5
PERCEPTUAL_LOSS_BETA = 1e-3

NUM_WORKERS = 8

TRAIN_DATASET_PATH = Path("data/DF2K_OST.txt")
VAL_DATASET_PATH = Path("data/DIV2K_valid.txt")
TEST_DATASETS_PATH = Path("data/test_datasets.txt")

SRRESNET_MODEL_CHECKPOINT_PATH = Path(
    "checkpoints/srresnet_best/srresnet_model_best.safetensors"
)
BEST_CHECKPOINT_DIR_PATH = Path("checkpoints/srgan_best")
CHECKPOINT_DIR_PATH = Path("checkpoints/srgan_latest")


def create_logger(
    log_level: str,
    caller_file_name: str,
    log_file_name: str | None = None,
    max_log_file_size: int = 5 * 1024 * 1024,
    backup_count: int = 10,
) -> logging.Logger:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y.%m.%d %H:%M:%S"
    )

    logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    logger.addHandler(console_handler)

    if not log_file_name:
        current_date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        log_file_name = f"logs/srgan_{Path(caller_file_name).stem}_{current_date}.log"

    log_file_path = Path(log_file_name)
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = RotatingFileHandler(
        filename=log_file_path,
        maxBytes=max_log_file_size,
        backupCount=backup_count,
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
