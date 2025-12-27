import argparse
from pathlib import Path

import fire
import torch

from tutorial_train import IPAdapterLitModule


@torch.inference_mode()
def export_ip_adapter_bin(ckpt: str, out_dir: str) -> None:
    ckpt = Path(ckpt)
    out_dir = Path(out_dir)

    lit = IPAdapterLitModule.load_from_checkpoint(str(ckpt), map_location="cpu")
    lit.eval()

    out_dir.mkdir(parents=True, exist_ok=True)
    lit.save_ip_adapter_weights(out_dir)
    print(f"[OK] Saved: {out_dir / 'ip_adapter.bin'}")


def main():
    fire.Fire(export_ip_adapter_bin)


if __name__ == "__main__":
    main()
