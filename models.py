import math
from typing import Literal

from torch import Tensor, nn
from torchvision.models import VGG19_Weights, vgg19
from torchvision.transforms import Normalize

from config import create_logger

logger = create_logger("INFO", __file__)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        norm_layer: bool = False,
        activation: Literal["prelu", "leaky_relu", "tanh"] | None = None,
    ) -> None:
        super().__init__()

        self.conv_block = nn.Sequential()

        self.conv_block.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
            )
        )

        if norm_layer:
            self.conv_block.append(nn.BatchNorm2d(out_channels))

        if activation:
            match activation.lower():
                case "prelu":
                    self.conv_block.append(nn.PReLU())
                case "leaky_relu":
                    self.conv_block.append(nn.LeakyReLU(0.2))
                case "tanh":
                    self.conv_block.append(nn.Tanh())

    def forward(self, x: Tensor) -> Tensor:
        return self.conv_block(x)


class SubPixelConvBlock(nn.Module):
    def __init__(
        self,
        channels_count: int,
        kernel_size: int,
        scaling_factor: Literal[2, 4, 8],
    ) -> None:
        super().__init__()

        self.subpixel_conv_block = nn.Sequential(
            nn.Conv2d(
                in_channels=channels_count,
                out_channels=channels_count * (scaling_factor**2),
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.PixelShuffle(upscale_factor=scaling_factor),
            nn.PReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.subpixel_conv_block(x)


class ResBlock(nn.Module):
    def __init__(self, channels_count: int, kernel_size: int) -> None:
        super().__init__()

        self.res_block = nn.Sequential(
            ConvBlock(
                in_channels=channels_count,
                out_channels=channels_count,
                kernel_size=kernel_size,
                norm_layer=True,
                activation="prelu",
            ),
            ConvBlock(
                in_channels=channels_count,
                out_channels=channels_count,
                kernel_size=kernel_size,
                norm_layer=True,
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.res_block(x) + x


class Generator(nn.Module):
    def __init__(
        self,
        channels_count: int,
        large_kernel_size: int,
        small_kernel_size: int,
        res_blocks_count: int,
        scaling_factor: Literal[2, 4, 8],
    ) -> None:
        super().__init__()

        self.conv_block1 = ConvBlock(
            in_channels=3,
            out_channels=channels_count,
            kernel_size=large_kernel_size,
            activation="prelu",
        )

        self.res_blocks = nn.Sequential(
            *[
                ResBlock(channels_count=channels_count, kernel_size=small_kernel_size)
                for _ in range(res_blocks_count)
            ]
        )

        self.conv_block2 = ConvBlock(
            in_channels=channels_count,
            out_channels=channels_count,
            kernel_size=small_kernel_size,
            norm_layer=True,
        )

        self.subpixel_conv_blocks = nn.Sequential(
            *[
                SubPixelConvBlock(
                    channels_count=channels_count,
                    kernel_size=small_kernel_size,
                    scaling_factor=2,
                )
                for _ in range(int(math.log2(scaling_factor)))
            ]
        )

        self.conv_block3 = ConvBlock(
            in_channels=channels_count,
            out_channels=3,
            kernel_size=large_kernel_size,
            activation="tanh",
        )

    def forward(self, x: Tensor) -> Tensor:
        output = self.conv_block1(x)
        residual = output
        output = self.res_blocks(output)
        output = self.conv_block2(output)
        output += residual
        output = self.subpixel_conv_blocks(output)
        output = self.conv_block3(output)

        return output


class Discriminator(nn.Module):
    def __init__(
        self,
        channels_count: int,
        kernel_size: int,
        conv_blocks_count: int,
        linear_layer_size: int,
    ) -> None:
        super().__init__()

        conv_blocks = []

        conv_blocks.append(
            ConvBlock(
                in_channels=3,
                out_channels=channels_count,
                kernel_size=kernel_size,
                activation="leaky_relu",
            )
        )

        in_channels = channels_count

        for i in range(1, conv_blocks_count):
            out_channels = in_channels * 2 if i % 2 == 0 else in_channels
            stride = 1 if i % 2 == 0 else 2

            conv_blocks.append(
                ConvBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    norm_layer=True,
                    activation="leaky_relu",
                )
            )

            in_channels = out_channels

        self.layers = nn.Sequential(
            *conv_blocks,
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Linear(in_channels * 6 * 6, linear_layer_size),
            nn.LeakyReLU(0.2),
            nn.Linear(linear_layer_size, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class TruncatedVGG19(nn.Module):
    def __init__(self):
        super().__init__()

        vgg19_model = vgg19(weights=VGG19_Weights.DEFAULT)

        self.features = nn.Sequential(*list(vgg19_model.features.children())[:36])
        self.normalization = Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        self.features.eval()
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x: Tensor) -> Tensor:
        x_rescaled = (x + 1.0) / 2.0
        x_normilized = self.normalization(x_rescaled)
        output = self.features(x_normilized)

        return output
