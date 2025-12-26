from __future__ import annotations

from pathlib import Path
import argparse
import importlib.util
import fire
import numpy as np
import torch

from tutorial_train import IPAdapterLitModule


class IPAdapterUnetStepForOnnx(torch.nn.Module):
    def __init__(self, unet: torch.nn.Module, image_proj_model: torch.nn.Module):
        super().__init__()
        self.unet = unet
        self.image_proj_model = image_proj_model

    def forward(self, noisy_latents, timesteps, encoder_hidden_states, image_embeds):
        ip_tokens = self.image_proj_model(image_embeds)
        cond = torch.cat([encoder_hidden_states, ip_tokens], dim=1)
        return self.unet(noisy_latents, timesteps, cond, return_dict=False)[0]


def validate(
    ckpt: str,
    onnx_path: str,
    device: str,
    resolution: int,
    batch: int,
) -> None:
    try:
        import onnxruntime as ort
    except Exception as e:
        raise RuntimeError('onnxruntime is not installed. Install: uv add onnxruntime') from e
    ckpt = Path(ckpt)
    onnx_path = Path(onnx_path)

    lit = IPAdapterLitModule.load_from_checkpoint(str(ckpt), map_location="cpu")
    lit.eval()

    dev = torch.device(device)
    lit.to(dev)

    model = IPAdapterUnetStepForOnnx(lit.unet, lit.image_proj_model).eval().to(dev)

    latent_h = resolution // 8
    latent_w = resolution // 8
    latent_c = int(getattr(lit.unet.config, "in_channels", 4))
    text_hidden = int(getattr(lit.text_encoder.config, "hidden_size", 768))
    max_len = int(getattr(lit.tokenizer, "model_max_length", 77))
    img_embed_dim = int(getattr(lit.image_encoder.config, "projection_dim", 1024))

    noisy_latents = torch.randn(batch, latent_c, latent_h, latent_w, device=dev, dtype=torch.float32)
    timesteps = torch.randint(0, 1000, (batch,), device=dev, dtype=torch.long)
    encoder_hidden_states = torch.randn(batch, max_len, text_hidden, device=dev, dtype=torch.float32)
    image_embeds = torch.randn(batch, img_embed_dim, device=dev, dtype=torch.float32)

    with torch.no_grad():
        y_pt = model(noisy_latents, timesteps, encoder_hidden_states, image_embeds).cpu().numpy()

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    y_onnx = sess.run(
        None,
        {
            "noisy_latents": noisy_latents.cpu().numpy(),
            "timesteps": timesteps.cpu().numpy(),
            "encoder_hidden_states": encoder_hidden_states.cpu().numpy(),
            "image_embeds": image_embeds.cpu().numpy(),
        },
    )[0]

    diff = np.abs(y_pt - y_onnx)
    print("[OK] Validation results")
    print("     max_abs_diff:", float(diff.max()))
    print("     mean_abs_diff:", float(diff.mean()))
    print("     y_pt mean/std:", float(y_pt.mean()), float(y_pt.std()))
    print("     y_onnx mean/std:", float(y_onnx.mean()), float(y_onnx.std()))


def main() -> None:
    fire.Fire(validate)


if __name__ == "__main__":
    main()
