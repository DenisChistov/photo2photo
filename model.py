from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import torch
from PIL import Image
from torch import nn


Direction = Literal["a_to_b", "b_to_a"]


class ResnetBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        norm_layer: type[nn.Module] = nn.InstanceNorm2d,
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, bias=use_bias),
            norm_layer(dim),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, bias=use_bias),
            norm_layer(dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ResnetGenerator(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        ngf: int = 64,
        n_res_blocks: int = 6,
        norm_layer: type[nn.Module] = nn.InstanceNorm2d,
    ) -> None:
        super().__init__()
        use_bias = True

        layers: list[nn.Module] = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(
                in_channels,
                ngf,
                kernel_size=7,
                bias=use_bias,
            ),
            norm_layer(ngf),
            nn.ReLU(inplace=True),
        ]

        n_down = 2
        for i in range(n_down):
            multiplier = 2**i
            layers.extend(
                [
                    nn.Conv2d(
                        ngf * multiplier,
                        ngf * multiplier * 2,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias=use_bias,
                    ),
                    norm_layer(ngf * multiplier * 2),
                    nn.ReLU(inplace=True),
                ]
            )

        multiplier = 2**n_down
        for _ in range(n_res_blocks):
            layers.append(
                ResnetBlock(
                    ngf * multiplier,
                    norm_layer=norm_layer,
                    use_bias=use_bias,
                )
            )

        for i in range(n_down):
            multiplier = 2 ** (n_down - i)
            layers.extend(
                [
                    nn.ConvTranspose2d(
                        ngf * multiplier,
                        ngf * multiplier // 2,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                        bias=use_bias,
                    ),
                    norm_layer(ngf * multiplier // 2),
                    nn.ReLU(inplace=True),
                ]
            )

        layers.extend(
            [
                nn.ReflectionPad2d(3),
                nn.Conv2d(
                    ngf,
                    out_channels,
                    kernel_size=7,
                ),
                nn.Tanh(),
            ]
        )

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


@dataclass(frozen=True)
class CycleGANInference:
    generator_a_to_b: ResnetGenerator
    generator_b_to_a: ResnetGenerator
    image_size: int
    mean_a: tuple[float, float, float]
    std_a: tuple[float, float, float]
    mean_b: tuple[float, float, float]
    std_b: tuple[float, float, float]
    dataset_name: str


def _triple(
    value: Sequence[float] | torch.Tensor,
    field_name: str,
) -> tuple[float, float, float]:
    items = tuple(float(item) for item in value)
    if len(items) != 3:
        raise ValueError(
            f"Поле {field_name!r} должно содержать три значения, получено: {items}"
        )
    return items


def _safe_torch_load(checkpoint_path: str | Path) -> dict:
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
        )

    if not isinstance(checkpoint, dict):
        raise TypeError("Чекпоинт должен содержать словарь параметров.")

    return checkpoint


def load_cyclegan_checkpoint(
    checkpoint_path: str | Path,
) -> CycleGANInference:
    checkpoint = _safe_torch_load(checkpoint_path)

    required_keys = {
        "gen_a_to_b",
        "gen_b_to_a",
        "model_params",
        "image_size",
    }
    missing = required_keys.difference(checkpoint)
    if missing:
        raise ValueError(
            "Это не инференс-чекпоинт CycleGAN. "
            f"Отсутствуют ключи: {sorted(missing)}. "
            "Используйте cyclegan_export.pt или cyclegan_export_monet.pt, "
            "а не файл *_full.pt."
        )

    params = checkpoint["model_params"]
    in_channels = int(params.get("in_channels", 3))
    ngf = int(params["ngf"])
    n_res_blocks = int(params["n_res_blocks"])

    generator_a_to_b = ResnetGenerator(
        in_channels=in_channels,
        out_channels=3,
        ngf=ngf,
        n_res_blocks=n_res_blocks,
    )
    generator_b_to_a = ResnetGenerator(
        in_channels=in_channels,
        out_channels=3,
        ngf=ngf,
        n_res_blocks=n_res_blocks,
    )

    generator_a_to_b.load_state_dict(
        checkpoint["gen_a_to_b"],
        strict=True,
    )
    generator_b_to_a.load_state_dict(
        checkpoint["gen_b_to_a"],
        strict=True,
    )

    generator_a_to_b.eval()
    generator_b_to_a.eval()

    for generator in (generator_a_to_b, generator_b_to_a):
        generator.requires_grad_(False)

    return CycleGANInference(
        generator_a_to_b=generator_a_to_b,
        generator_b_to_a=generator_b_to_a,
        image_size=int(checkpoint["image_size"]),
        mean_a=_triple(
            checkpoint.get("mean_a", (0.5, 0.5, 0.5)),
            "mean_a",
        ),
        std_a=_triple(
            checkpoint.get("std_a", (0.5, 0.5, 0.5)),
            "std_a",
        ),
        mean_b=_triple(
            checkpoint.get("mean_b", (0.5, 0.5, 0.5)),
            "mean_b",
        ),
        std_b=_triple(
            checkpoint.get("std_b", (0.5, 0.5, 0.5)),
            "std_b",
        ),
        dataset_name=str(checkpoint.get("dataset_name", "cyclegan")),
    )


def _preprocess(
    image: Image.Image,
    image_size: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> torch.Tensor:
    resized = image.convert("RGB").resize(
        (image_size, image_size),
        resample=Image.Resampling.BICUBIC,
    )

    array = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)

    mean_tensor = torch.tensor(
        mean,
        dtype=tensor.dtype,
    ).view(1, 3, 1, 1)
    std_tensor = torch.tensor(
        std,
        dtype=tensor.dtype,
    ).view(1, 3, 1, 1)

    return (tensor - mean_tensor) / std_tensor


def _postprocess(
    tensor: torch.Tensor,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> Image.Image:
    tensor = tensor.detach().cpu()

    mean_tensor = torch.tensor(
        mean,
        dtype=tensor.dtype,
    ).view(1, 3, 1, 1)
    std_tensor = torch.tensor(
        std,
        dtype=tensor.dtype,
    ).view(1, 3, 1, 1)

    tensor = tensor * std_tensor + mean_tensor
    tensor = tensor.clamp(0.0, 1.0)[0]
    array = (
        tensor.permute(1, 2, 0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .numpy()
    )
    return Image.fromarray(array, mode="RGB")


@torch.inference_mode()
def translate_image(
    image: Image.Image,
    model: CycleGANInference,
    direction: Direction,
) -> Image.Image:
    if direction == "a_to_b":
        generator = model.generator_a_to_b
        input_mean = model.mean_a
        input_std = model.std_a
        output_mean = model.mean_b
        output_std = model.std_b
    elif direction == "b_to_a":
        generator = model.generator_b_to_a
        input_mean = model.mean_b
        input_std = model.std_b
        output_mean = model.mean_a
        output_std = model.std_a
    else:
        raise ValueError(
            "direction должен быть равен 'a_to_b' или 'b_to_a'."
        )

    input_tensor = _preprocess(
        image=image,
        image_size=model.image_size,
        mean=input_mean,
        std=input_std,
    )
    output_tensor = generator(input_tensor)

    return _postprocess(
        tensor=output_tensor,
        mean=output_mean,
        std=output_std,
    )
