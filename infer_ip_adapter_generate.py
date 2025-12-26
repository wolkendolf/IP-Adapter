from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import fire
import torch
from PIL import Image

from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL

from ip_adapter.ip_adapter import IPAdapter


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def _safe_stem(s: str) -> str:
    keep = []
    for ch in s:
        if ch.isalnum() or ch in ("-", "_", "."):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep)[:120]


def _save_images(images: list[Image.Image], out_dir: Path, prefix: str) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for i, img in enumerate(images):
        p = out_dir / f"{prefix}_{i:02d}.png"
        img.save(p)
        paths.append(str(p))
    return paths


@torch.inference_mode()
def run(
    ip_ckpt: str,
    image_encoder_path: str,
    out_dir: str = "outputs/infer",
    # base model
    base_model: str = "runwayml/stable-diffusion-v1-5",
    vae_model: str | None = None,
    # inputs
    requests_jsonl: str | None = None,
    image: str | None = None,
    prompt: str | None = None,
    negative_prompt: str | None = None,
    # generation params (defaults)
    num_samples: int = 4,
    seed: int | None = 42,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 30,
    scale: float = 1.0,
    num_tokens: int = 4,
    device: str = "cuda",
) -> str:
    """
    Generate images using trained IP-Adapter weights.

    Usage:
      - Batch mode: provide requests_jsonl
      - Single mode: provide image + prompt

    Args:
        ip_ckpt: Path to trained weights (ip_adapter.bin).
        image_encoder_path: Path or HF-id to CLIP vision encoder directory used in train.
        out_dir: Output directory for generated images + manifest.jsonl
        base_model: HF-id or local path to Stable Diffusion model
        vae_model: Optional HF-id/local path for VAE
        requests_jsonl: JSONL file with inference requests
        image/prompt: single-request mode
        negative_prompt: optional
        num_samples, seed, guidance_scale, num_inference_steps, scale: generation parameters
        num_tokens: number of IP tokens (must match training)
        device: "cuda" or "cpu"
    Returns:
        Path to manifest.jsonl
    """
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    # dtype: fp16 only makes sense on cuda
    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.float16
    else:
        device = "cpu"
        dtype = torch.float32

    # Scheduler (как в официальных примерах IP-Adapter часто используют DDIM)
    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )

    # Optional VAE
    vae = None
    if vae_model:
        vae = AutoencoderKL.from_pretrained(vae_model).to(dtype=dtype)

    # Load SD pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model,
        torch_dtype=dtype,
        scheduler=noise_scheduler,
        vae=vae,
        feature_extractor=None,
        safety_checker=None,
    ).to(device)

    # Load IP-Adapter
    ip_model = IPAdapter(
        sd_pipe=pipe,
        image_encoder_path=image_encoder_path,
        ip_ckpt=ip_ckpt,
        device=device,
        num_tokens=num_tokens,
        dtype=dtype,
    )

    # Prepare requests
    if requests_jsonl is not None:
        reqs = _read_jsonl(Path(requests_jsonl))
        if not reqs:
            raise ValueError(f"requests_jsonl is empty: {requests_jsonl}")
    else:
        if image is None or prompt is None:
            raise ValueError("Provide either --requests_jsonl or both --image and --prompt")
        reqs = [{
            "id": "single",
            "image": image,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_samples": num_samples,
            "seed": seed,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "scale": scale,
        }]

    manifest_path = out_dir_p / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as mf:
        for idx, r in enumerate(reqs):
            img_path = Path(r["image"])
            if not img_path.exists():
                raise FileNotFoundError(f"Image not found: {img_path}")

            rid = str(r.get("id", f"item_{idx:05d}"))
            prefix = _safe_stem(rid)

            r_prompt = r.get("prompt", prompt) or "best quality, high quality"
            r_neg = r.get("negative_prompt", negative_prompt) or "monochrome, lowres, bad anatomy, worst quality, low quality"

            r_num_samples = int(r.get("num_samples", num_samples))
            r_seed = r.get("seed", seed)
            r_guidance = float(r.get("guidance_scale", guidance_scale))
            r_steps = int(r.get("num_inference_steps", num_inference_steps))
            r_scale = float(r.get("scale", scale))

            pil = Image.open(img_path).convert("RGB")

            images = ip_model.generate(
                pil_image=pil,
                prompt=r_prompt,
                negative_prompt=r_neg,
                scale=r_scale,
                num_samples=r_num_samples,
                seed=r_seed,
                guidance_scale=r_guidance,
                num_inference_steps=r_steps,
            )

            saved = _save_images(images, out_dir_p, prefix)

            out_rec = {
                "id": rid,
                "image": str(img_path),
                "prompt": r_prompt,
                "negative_prompt": r_neg,
                "num_samples": r_num_samples,
                "seed": r_seed,
                "guidance_scale": r_guidance,
                "num_inference_steps": r_steps,
                "scale": r_scale,
                "outputs": saved,
            }
            mf.write(json.dumps(out_rec, ensure_ascii=False) + "\n")

            print(f"[OK] {rid}: saved {len(saved)} images")

    print(f"[OK] Manifest: {manifest_path}")
    return str(manifest_path)


if __name__ == "__main__":
    fire.Fire(run)
