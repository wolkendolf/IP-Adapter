from __future__ import annotations

import argparse
import importlib.util
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import fire
import torch

from tutorial_train import IPAdapterLitModule


@dataclass
class ExportMeta:
    opset: int
    device: str
    latent_shape: list[int]
    encoder_hidden_states_shape: list[int]
    image_embeds_shape: list[int]
    output_shape: list[int]


class IPAdapterUnetStepForOnnx(torch.nn.Module):
    def __init__(self, unet: torch.nn.Module, image_proj_model: torch.nn.Module):
        super().__init__()
        self.unet = unet
        self.image_proj_model = image_proj_model

    def forward(self, noisy_latents, timesteps, encoder_hidden_states, image_embeds):
        ip_tokens = self.image_proj_model(image_embeds)
        cond = torch.cat([encoder_hidden_states, ip_tokens], dim=1)
        return self.unet(noisy_latents, timesteps, cond, return_dict=False)[0]


def export_onnx(
    lightning_ckpt: str,
    out_onnx: str,
    opset: int,
    device: str,
    resolution: int,
    batch: int,
) -> None:
    lightning_ckpt = Path(lightning_ckpt)
    out_onnx = Path(out_onnx)
    # Load Lightning module (it will reconstruct from saved hparams)
    lit = IPAdapterLitModule.load_from_checkpoint(
        str(lightning_ckpt), map_location="cpu"
    )
    lit.eval()

    # Move to device
    dev = torch.device(device)
    lit.to(dev)

    # Build ONNX export model (single diffusion step)
    model = (
        IPAdapterUnetStepForOnnx(
            unet=lit.unet,
            image_proj_model=lit.image_proj_model,
        )
        .eval()
        .to(dev)
    )

    # 4) Determine dims from loaded components
    # SD latents spatial = resolution / 8
    latent_h = resolution // 8
    latent_w = resolution // 8
    latent_c = int(getattr(lit.unet.config, "in_channels", 4))

    # text hidden states
    text_hidden = int(getattr(lit.text_encoder.config, "hidden_size", 768))
    max_len = int(getattr(lit.tokenizer, "model_max_length", 77))

    # image embeds dim
    img_embed_dim = int(getattr(lit.image_encoder.config, "projection_dim", 1024))

    # Dummy inputs
    noisy_latents = torch.randn(
        batch, latent_c, latent_h, latent_w, device=dev, dtype=torch.float32
    )
    timesteps = torch.randint(0, 1000, (batch,), device=dev, dtype=torch.long)
    encoder_hidden_states = torch.randn(
        batch, max_len, text_hidden, device=dev, dtype=torch.float32
    )
    image_embeds = torch.randn(batch, img_embed_dim, device=dev, dtype=torch.float32)

    # Export
    out_onnx.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        (noisy_latents, timesteps, encoder_hidden_states, image_embeds),
        f=str(out_onnx),
        opset_version=opset,
        export_params=True,
        do_constant_folding=True,
        input_names=[
            "noisy_latents",
            "timesteps",
            "encoder_hidden_states",
            "image_embeds",
        ],
        output_names=["noise_pred"],
        dynamic_axes={
            "noisy_latents": {0: "batch", 2: "latent_h", 3: "latent_w"},
            "timesteps": {0: "batch"},
            "encoder_hidden_states": {0: "batch"},
            "image_embeds": {0: "batch"},
            "noise_pred": {0: "batch", 2: "latent_h", 3: "latent_w"},
        },
    )

    # Save meta info (useful for README / inference)
    with torch.no_grad():
        y = model(noisy_latents, timesteps, encoder_hidden_states, image_embeds)

    meta = ExportMeta(
        opset=opset,
        device=device,
        latent_shape=list(noisy_latents.shape),
        encoder_hidden_states_shape=list(encoder_hidden_states.shape),
        image_embeds_shape=list(image_embeds.shape),
        output_shape=list(y.shape),
    )
    meta_path = out_onnx.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(asdict(meta), indent=2), encoding="utf-8")

    print(f"[OK] Saved ONNX: {out_onnx}")
    print(f"[OK] Saved meta: {meta_path}")
    print(f"     output shape: {list(y.shape)}")


def main() -> None:
    fire.Fire(export_onnx)


if __name__ == "__main__":
    main()
