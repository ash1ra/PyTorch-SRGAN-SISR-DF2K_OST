from pathlib import Path
from typing import Literal

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from config import create_logger
from utils import transform_image

logger = create_logger("INFO", __file__)


class SRDataset(Dataset):
    def __init__(
        self,
        data_path: str | Path,
        scaling_factor: Literal[2, 4, 8],
        crop_size: int | None = None,
        test_mode: bool = False,
        dev_mode: bool = False,
    ) -> None:
        self.scaling_factor = scaling_factor
        self.crop_size = crop_size
        self.test_mode = test_mode
        self.images = []

        data_path = Path(data_path)

        if not data_path.exists():
            logger.error(f'Path does not exists: "{data_path}"')
            raise FileNotFoundError

        if data_path.is_dir():
            logger.info(f'Creating dataset from directory ("{data_path}")...')
            image_paths = list(data_path.iterdir())
        elif data_path.is_file():
            logger.info(f'Creating dataset from file ("{data_path}")...')
            try:
                with open(data_path, "r") as f:
                    image_paths = [Path(line.strip()) for line in f.readlines() if line]
            except FileNotFoundError:
                logger.error(f'File with images list was not found ("{data_path}")')

        try:
            for image_path in image_paths:
                if image_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    with Image.open(image_path) as img:
                        if img.mode == "RGB":
                            if test_mode:
                                self.images.append(image_path)
                            else:
                                width, height = img.size
                                if width >= self.crop_size and height >= self.crop_size:
                                    self.images.append(image_path)
        except FileNotFoundError:
            logger.error(f'Image at path "{image_path}" was not found, skipping...')

        if dev_mode:
            self.images = self.images[: int(len(self.images) * 0.1)]

        print(type(self.images[342]))

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int) -> tuple[Tensor, Tensor]:
        return transform_image(
            image_path=self.images[i],
            scaling_factor=self.scaling_factor,
            crop_size=self.crop_size,
            test_mode=self.test_mode,
        )
