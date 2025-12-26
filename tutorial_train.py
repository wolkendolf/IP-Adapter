import os
import random
import argparse
from pathlib import Path
import json
import itertools
import time
from typing import Optional, Dict, Any
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field

import hydra
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig, OmegaConf

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from lightning.pytorch.loggers import MLFlowLogger

import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from transformers import CLIPImageProcessor
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

from ip_adapter.ip_adapter import ImageProjModel
from ip_adapter.utils import is_torch2_available
if is_torch2_available():
    from ip_adapter.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor
else:
    from ip_adapter.attention_processor import IPAttnProcessor, AttnProcessor

from tools.data_access import ensure_training_data


# Dataset
class MyDataset(torch.utils.data.Dataset):

    def __init__(self, json_file, tokenizer, size=512, t_drop_rate=0.05, i_drop_rate=0.05, ti_drop_rate=0.05, image_root_path=""):
        super().__init__()

        self.tokenizer = tokenizer
        self.size = size
        self.i_drop_rate = i_drop_rate
        self.t_drop_rate = t_drop_rate
        self.ti_drop_rate = ti_drop_rate
        self.image_root_path = image_root_path

        with open(json_file, "r", encoding="utf-8") as f:
            # list of dict: [{"image_file": "1.png", "text": "A dog"}]
            self.data = json.load(f)

        self.transform = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.clip_image_processor = CLIPImageProcessor()
        
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]
        image_file = item["image_file"]
        
        # read image
        raw_image = Image.open(os.path.join(self.image_root_path, image_file))
        image = self.transform(raw_image.convert("RGB"))
        clip_image = self.clip_image_processor(images=raw_image, return_tensors="pt").pixel_values
        
        # drop
        drop_image_embed = 0
        rand_num = random.random()
        if rand_num < self.i_drop_rate:
            drop_image_embed = 1
        elif rand_num < (self.i_drop_rate + self.t_drop_rate):
            text = ""
        elif rand_num < (self.i_drop_rate + self.t_drop_rate + self.ti_drop_rate):
            text = ""
            drop_image_embed = 1
        # get text and tokenize
        text_input_ids = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        
        return {
            "image": image,
            "text_input_ids": text_input_ids,
            "clip_image": clip_image,
            "drop_image_embed": drop_image_embed
        }

    def __len__(self):
        return len(self.data)
    

def collate_fn(data):
    images = torch.stack([example["image"] for example in data])
    text_input_ids = torch.cat([example["text_input_ids"] for example in data], dim=0)
    clip_images = torch.cat([example["clip_image"] for example in data], dim=0)
    drop_image_embeds = torch.tensor([example["drop_image_embed"] for example in data], dtype=torch.bool)

    return {
        "images": images,
        "text_input_ids": text_input_ids,
        "clip_images": clip_images,
        "drop_image_embeds": drop_image_embeds
    }
    
class IPAdapterDataModule(L.LightningDataModule):
    def __init__(
        self,
        *,
        data_json_file: str,
        data_root_path: str,
        tokenizer: CLIPTokenizer,
        resolution: int,
        train_batch_size: int,
        num_workers: int,
    ) -> None:
        super().__init__()
        self.data_json_file = data_json_file
        self.data_root_path = data_root_path
        self.tokenizer = tokenizer
        self.resolution = resolution
        self.train_batch_size = train_batch_size
        self.num_workers = num_workers
        self.train_dataset: Optional[MyDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = MyDataset(
            self.data_json_file,
            tokenizer=self.tokenizer,
            size=self.resolution,
            image_root_path=self.data_root_path,
        )

    def train_dataloader(self) -> DataLoader:
        assert self.train_dataset is not None
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

class IPAdapter(torch.nn.Module):
    """IP-Adapter"""
    def __init__(self, unet, image_proj_model, adapter_modules, ckpt_path=None):
        super().__init__()
        self.unet = unet
        self.image_proj_model = image_proj_model
        self.adapter_modules = adapter_modules

        if ckpt_path is not None:
            self.load_from_checkpoint(ckpt_path)

    def forward(self, noisy_latents, timesteps, encoder_hidden_states, image_embeds):
        ip_tokens = self.image_proj_model(image_embeds)
        encoder_hidden_states = torch.cat([encoder_hidden_states, ip_tokens], dim=1)
        # Predict the noise residual
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
        return noise_pred

    def load_from_checkpoint(self, ckpt_path: str):
        # Calculate original checksums
        orig_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        orig_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        state_dict = torch.load(ckpt_path, map_location="cpu")

        # Load state dict for image_proj_model and adapter_modules
        self.image_proj_model.load_state_dict(state_dict["image_proj"], strict=True)
        self.adapter_modules.load_state_dict(state_dict["ip_adapter"], strict=True)

        # Calculate new checksums
        new_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        new_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        # Verify if the weights have changed
        assert orig_ip_proj_sum != new_ip_proj_sum, "Weights of image_proj_model did not change!"
        assert orig_adapter_sum != new_adapter_sum, "Weights of adapter_modules did not change!"

        print(f"Successfully loaded weights from checkpoint {ckpt_path}")

class IPAdapterLitModule(L.LightningModule):
    def __init__(
        self,
        *,
        pretrained_model_name_or_path: str,
        image_encoder_path: str,
        pretrained_ip_adapter_path: Optional[str],
        learning_rate: float,
        weight_decay: float,
        num_tokens: int = 4,
        save_ip_adapter_every_n_steps: int = 0,
        output_dir: str = "sd-ip_adapter",
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore="model") # ignore="model" чтобы не занимать много времени на сохранение
        
        # ---- Load scheduler/tokenizer/models ----
        self.noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
        self.tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")

        self.text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder")
        self.vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet")
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder_path)

        # ---- Freeze big components (как у тебя) ----
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.image_encoder.requires_grad_(False)

        # ---- IP-Adapter image proj ----
        self.image_proj_model = ImageProjModel(
            cross_attention_dim=self.unet.config.cross_attention_dim,
            clip_embeddings_dim=self.image_encoder.config.projection_dim,
            clip_extra_context_tokens=num_tokens,
        )

        # ---- Init adapter modules (attention processors) ----
        attn_procs: Dict[str, torch.nn.Module] = {}
        unet_sd = self.unet.state_dict()
        for name in self.unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else self.unet.config.cross_attention_dim

            if name.startswith("mid_block"):
                hidden_size = self.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet.config.block_out_channels[block_id]
            else:
                raise ValueError(f"Unknown attention processor name: {name}")

            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor()
            else:
                layer_name = name.split(".processor")[0]
                weights = {
                    "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                    "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
                }
                proc = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
                proc.load_state_dict(weights, strict=False)
                attn_procs[name] = proc
        self.unet.set_attn_processor(attn_procs)
        self.adapter_modules = torch.nn.ModuleList(self.unet.attn_processors.values())

        self.ip_adapter = IPAdapter(
            self.unet,
            self.image_proj_model,
            self.adapter_modules,
            ckpt_path=pretrained_ip_adapter_path,
        )

        self.vae.eval()
        self.text_encoder.eval()
        self.image_encoder.eval()
        self.unet.eval()

        # where to dump standalone ip-adapter weights
        self._output_dir = Path(output_dir)
    
    def _weight_dtype(self) -> torch.dtype:
        # Lightning управляет autocast сам, но нам удобно приводить inputs как в original
        # (иначе CLIP/VAE могут работать в fp32 и тратить память).
        prec = str(self.trainer.precision) if self.trainer is not None else "32"
        if "bf16" in prec:
            return torch.bfloat16
        if "16" in prec:
            return torch.float16
        return torch.float32
    
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        weight_dtype = self._weight_dtype()

        # VAE encode -> latents
        with torch.no_grad():
            latents = self.vae.encode(batch["images"].to(self.device, dtype=weight_dtype)).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor

        # diffusion noise
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(
            0, self.noise_scheduler.num_train_timesteps, (bsz,), device=latents.device, dtype=torch.long
        )
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # image embeds
        with torch.no_grad():
            image_embeds = self.image_encoder(batch["clip_images"].to(self.device, dtype=weight_dtype)).image_embeds

        # apply drop_image_embeds mask (vectorized)
        drop_mask = batch["drop_image_embeds"].to(self.device)  # bool (B,)
        if drop_mask.any():
            image_embeds = image_embeds.clone()
            image_embeds[drop_mask] = 0

        # text embeds
        with torch.no_grad():
            encoder_hidden_states = self.text_encoder(batch["text_input_ids"].to(self.device))[0]

        noise_pred = self.ip_adapter(noisy_latents, timesteps, encoder_hidden_states, image_embeds)
        loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

        # Логи (Lightning сам агрегирует/синхронизирует при DDP если надо)
        drop_frac = drop_mask.float().mean()
        noise_pred_std = noise_pred.float().std()
        timestep_mean = timesteps.float().mean()

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/drop_frac", drop_frac, on_step=True, on_epoch=True, prog_bar=False)
        self.log("train/noise_pred_std", noise_pred_std, on_step=True, on_epoch=True, prog_bar=False)
        self.log("train/timestep_mean", timestep_mean, on_step=True, on_epoch=True, prog_bar=False)

        return loss
    
    def configure_optimizers(self):
        params_to_opt = itertools.chain(
            self.ip_adapter.image_proj_model.parameters(),
            self.ip_adapter.adapter_modules.parameters(),
        )
        return torch.optim.AdamW(params_to_opt, lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
    
    @torch.inference_mode()
    def save_ip_adapter_weights(self, save_dir: Path) -> None:
        save_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "image_proj": self.ip_adapter.image_proj_model.state_dict(),
            "ip_adapter": self.ip_adapter.adapter_modules.state_dict(),
        }
        torch.save(state, save_dir / "ip_adapter.bin")
    
    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        # Дополнительно к lightning-checkpoint сохраняем совместимый ip_adapter.bin каждые N шагов.
        n = int(self.hparams.save_ip_adapter_every_n_steps or 0)
        if n <= 0:
            return
        step = int(self.global_step)
        if step > 0 and step % n == 0 and self.trainer.is_global_zero:
            self.save_ip_adapter_weights(self._output_dir / f"checkpoint-{step}")


def ensure_dvc_data(path: Path, remote: str = "data", jobs: int = 1) -> None:
    # Если данные уже на месте — DVC быстро проверит и ничего не скачает
    cmd = ["dvc", "pull", str(path), "-r", remote, "-j", str(jobs)]
    subprocess.run(cmd, check=True)

def _to_float(x):
    if x is None:
        return None
    try:
        return float(x.detach().cpu().item())
    except Exception:
        try:
            return float(x)
        except Exception:
            return None


def _get_mlflow_logger(trainer) -> MLFlowLogger | None:
    for lg in getattr(trainer, "loggers", []) or []:
        if isinstance(lg, MLFlowLogger):
            return lg
    # fallback
    lg = getattr(trainer, "logger", None)
    if isinstance(lg, MLFlowLogger):
        return lg
    return None

@dataclass
class ExperimentMetaCallback(Callback):
    cfg: Any
    plots_dir: Path
    repo_root: Path

    def on_fit_start(self, trainer, pl_module) -> None:
        meta_dir = self.plots_dir / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)

        # полный config как artifact (самый надёжный способ сохранить все гиперпараметры)
        cfg_yaml = OmegaConf.to_yaml(self.cfg, resolve=True)
        (meta_dir / "config.yaml").write_text(cfg_yaml, encoding="utf-8")

        # 3) “основные” гиперпараметры как MLflow params (короткие поля)
        params = {
            "seed": str(getattr(self.cfg, "seed", "")),
            "lr": str(self.cfg.train.learning_rate),
            "weight_decay": str(self.cfg.train.weight_decay),
            "batch_size": str(self.cfg.data.train_batch_size),
            "resolution": str(self.cfg.data.resolution),
            "num_tokens": str(self.cfg.model.num_tokens),
            "precision": str(self.cfg.trainer.precision),
            "max_epochs": str(self.cfg.trainer.max_epochs),
        }

        mlflow_logger = _get_mlflow_logger(trainer)
        if mlflow_logger is not None:
            mlflow_logger.log_hyperparams(params)

            # логируем файлы как artifacts
            mlflow_logger.experiment.log_artifact(mlflow_logger.run_id, str(meta_dir / "config.yaml"))


@dataclass
class MetricsPlotCallback(Callback):
    plots_dir: Path
    metrics_to_plot: tuple[str, ...] = (
        "train/loss_step",
        "train/noise_pred_std_step",
        "lr-AdamW",  # имя обычно такое; если будет иначе — всё равно поймаем по префиксу lr-
    )
    steps: list[int] = field(default_factory=list)
    history: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))

    def _read_metric(self, trainer, key: str):
        for d in (trainer.callback_metrics, trainer.logged_metrics, trainer.progress_bar_metrics):
            if key in d:
                return d.get(key)
        return None

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        step = int(trainer.global_step)
        self.steps.append(step)

        # если lr-AdamW окажется другим, попробуем найти любой ключ начинающийся с "lr-"
        lr_key = None
        for d in (trainer.callback_metrics, trainer.logged_metrics, trainer.progress_bar_metrics):
            for k in d.keys():
                if isinstance(k, str) and k.startswith("lr-"):
                    lr_key = k
                    break
            if lr_key:
                break

        keys = list(self.metrics_to_plot)
        if lr_key and lr_key not in keys:
            keys.append(lr_key)

        for k in keys:
            v = _to_float(self._read_metric(trainer, k))
            # сохраняем NaN если метрики на шаге нет, чтобы длины совпадали
            self.history[k].append(float("nan") if v is None else v)

    def on_fit_end(self, trainer, pl_module) -> None:
        out_dir = self.plots_dir / "metrics"
        out_dir.mkdir(parents=True, exist_ok=True)

        mlflow_logger = _get_mlflow_logger(trainer)

        for name, values in self.history.items():
            # Требуем минимум 2 точки чтобы строить
            if len(values) < 2:
                continue

            fig = plt.figure()
            plt.plot(self.steps, values)
            plt.xlabel("global_step")
            plt.ylabel(name)
            plt.title(name)
            png_path = out_dir / f"{name.replace('/', '_')}.png"
            fig.savefig(png_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

            if mlflow_logger is not None:
                mlflow_logger.experiment.log_artifact(mlflow_logger.run_id, str(png_path))

@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    # UNCOMMENT
    # ensure_training_data(cfg)

    L.seed_everything(int(cfg.seed), workers=True)

    repo_root = Path(to_absolute_path("."))
    plots_dir = repo_root / str(cfg.paths.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mlflow_logger = MLFlowLogger(
        tracking_uri=str(cfg.mlflow.tracking_uri),
        experiment_name=str(cfg.mlflow.experiment_name),
        run_name=str(cfg.mlflow.run_name),
    )

    ckpt_cb = ModelCheckpoint(
        dirpath=str(output_dir / "lightning_checkpoints"),
        every_n_train_steps=int(cfg.train.save_steps),
        save_last=True,
        save_top_k=-1,
        filename="step-{step}",
    )

    lr_cb = LearningRateMonitor(logging_interval="step")
    meta_cb = ExperimentMetaCallback(cfg=cfg, plots_dir=plots_dir, repo_root=repo_root)
    plot_cb = MetricsPlotCallback(plots_dir=plots_dir)

    model = IPAdapterLitModule(
        pretrained_model_name_or_path=str(cfg.model.pretrained_model_name_or_path),
        image_encoder_path=str(cfg.model.image_encoder_path),
        pretrained_ip_adapter_path=cfg.model.pretrained_ip_adapter_path,
        learning_rate=float(cfg.train.learning_rate),
        weight_decay=float(cfg.train.weight_decay),
        num_tokens=int(cfg.model.num_tokens),
        save_ip_adapter_every_n_steps=int(cfg.train.save_steps),
        output_dir=str(output_dir),
    )

    dm = IPAdapterDataModule(
        data_json_file=to_absolute_path(cfg.data.json_file),
        data_root_path=to_absolute_path(cfg.data.root_path),
        tokenizer=model.tokenizer,
        resolution=int(cfg.data.resolution),
        train_batch_size=int(cfg.data.train_batch_size),
        num_workers=int(cfg.data.num_workers),
    )

    trainer = L.Trainer(
        accelerator=str(cfg.trainer.accelerator),
        devices=int(cfg.trainer.devices),
        strategy=str(cfg.trainer.strategy),
        precision=str(cfg.trainer.precision),
        max_epochs=int(cfg.trainer.max_epochs),
        accumulate_grad_batches=int(cfg.trainer.accumulate_grad_batches),
        logger=mlflow_logger,
        callbacks=[ckpt_cb, lr_cb, meta_cb, plot_cb],
        log_every_n_steps=int(cfg.train.log_every_n_steps),
        enable_checkpointing=True,
    )

    trainer.fit(model, datamodule=dm)

if __name__ == "__main__":
    main()
