from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence

from omegaconf import DictConfig, OmegaConf


def _project_root() -> Path:
    """
    Hydra usually changes CWD to outputs/... .
    We want all relative paths from the original project root.
    """
    try:
        from hydra.utils import get_original_cwd  # type: ignore

        return Path(get_original_cwd())
    except Exception:
        return Path.cwd()


def _resolve(p: str | Path) -> Path:
    p = Path(p)
    if p.is_absolute():
        return p
    return _project_root() / p


def _as_rel_under_root(p: Path) -> str:
    root = _project_root().resolve()
    p = p.resolve()
    try:
        return str(p.relative_to(root))
    except Exception:
        # DVC usually expects paths inside the repo; fallback to absolute.
        return str(p)


# Subprocess helpers
def _run(cmd: list[str], *, cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def _require_exe(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"Executable '{name}' not found in PATH.")


def _find_tool_script(filename: str) -> Path:
    """
    Locate tools/*.py scripts regardless of Hydra run dir.
    """
    root = _project_root()
    candidates = [
        root / "tools" / filename,
        root / filename,
        Path(__file__).resolve().parent / filename,
        Path(__file__).resolve().parent.parent / "tools" / filename,
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"Cannot find '{filename}'. Tried: " + ", ".join(str(c) for c in candidates)
    )


# Checks
def _dir_nonempty(d: Path) -> bool:
    if not d.exists() or not d.is_dir():
        return False
    try:
        next(d.iterdir())
        return True
    except StopIteration:
        return False


def _final_dataset_ready(cfg: DictConfig) -> bool:
    """
    Checks that training dataset (cfg.data.json_file + cfg.data.root_path) exists.
    """
    json_file = _resolve(cfg.data.json_file)
    root_path = _resolve(cfg.data.root_path)
    return json_file.exists() and root_path.exists() and _dir_nonempty(root_path)


def _have_required_shards(processed_dir: Path, num_shards: int) -> bool:
    # require exactly the range [00000, ..., 000{num_shards-1}]
    for i in range(num_shards):
        if not (processed_dir / f"{i:05d}.tar").exists():
            return False
    return True


# DVC helpers (best-effort)
def _try_dvc_pull(processed_dir: Path, remote: str, jobs: int) -> bool:
    if shutil.which("dvc") is None:
        return False
    try:
        _run(["dvc", "pull", str(processed_dir), "-r", remote, "-j", str(jobs)])
        return True
    except subprocess.CalledProcessError:
        return False


# COYO WDS build: HF parquet -> subset parquet -> img2dataset (webdataset shards)
def _iter_parts(start: int, end: int) -> Sequence[int]:
    if end < start:
        raise ValueError(f"parquet_parts.end ({end}) < start ({start})")
    return list(range(start, end + 1))


# Step 1: Download parquet parts from HF (via wget)
def download_coyo_parquets(cfg: DictConfig) -> list[Path]:
    """
    Download HF parquet parts into cfg.data.coyo.raw_dir using wget.
    """
    _require_exe("wget")

    raw_dir = _resolve(cfg.data.coyo.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    parts = _iter_parts(
        int(cfg.data.coyo.parquet_parts.start), int(cfg.data.coyo.parquet_parts.end)
    )
    tpl = str(cfg.data.coyo.hf_url_template)

    out: list[Path] = []
    for part in parts:
        url = tpl.format(part=part)
        dst = raw_dir / Path(url).name
        out.append(dst)

        if dst.exists() and dst.stat().st_size > 0:
            continue

        _run(["wget", "-c", url, "-O", str(dst)])
    return out


# Step 2: Build a subset parquet
def make_subset_parquet(
    parquet_files: Sequence[Path],
    subset_path: Path,
    *,
    max_samples: int,
    columns: Sequence[str],
) -> None:
    """
    Build a small subset parquet with exactly max_samples rows (streaming).
    """
    try:
        import pyarrow as pa  # noqa: F401
        import pyarrow.parquet as pq
    except Exception as e:
        raise RuntimeError("pyarrow is required. Install: pip install pyarrow") from e

    subset_path.parent.mkdir(parents=True, exist_ok=True)
    if subset_path.exists():
        subset_path.unlink()

    writer: pq.ParquetWriter | None = None
    written = 0

    for f in parquet_files:
        pf = pq.ParquetFile(str(f))
        for batch in pf.iter_batches(batch_size=50_000, columns=list(columns)):
            if written >= max_samples:
                break

            remaining = max_samples - written
            if batch.num_rows > remaining:
                batch = batch.slice(0, remaining)

            table = pa.Table.from_batches([batch])

            if writer is None:
                writer = pq.ParquetWriter(str(subset_path), table.schema)

            writer.write_table(table)
            written += table.num_rows

        if written >= max_samples:
            break

    if writer is not None:
        writer.close()

    if written < max_samples:
        raise RuntimeError(
            f"Subset parquet has only {written} rows, expected {max_samples}. "
            f"Download more parquet parts (increase cfg.data.coyo.parquet_parts.end)."
        )


# Step 3: Run img2dataset
def run_img2dataset(cfg: DictConfig, subset_parquet: Path) -> None:
    _require_exe("img2dataset")

    wds_dir = _resolve(cfg.data.coyo.processed_dir)
    wds_dir.mkdir(parents=True, exist_ok=True)

    add_cols = list(cfg.data.coyo.img2dataset.additional_columns)
    add_cols_json = json.dumps(add_cols, ensure_ascii=False)

    cmd = [
        "img2dataset",
        "--url_list",
        str(subset_parquet),
        "--input_format",
        "parquet",
        "--url_col",
        "url",
        "--caption_col",
        "text",
        "--output_format",
        "webdataset",
        "--output_folder",
        str(wds_dir),
        "--number_sample_per_shard",
        str(int(cfg.data.coyo.samples_per_shard)),
        "--processes_count",
        str(int(cfg.data.coyo.img2dataset.processes_count)),
        "--thread_count",
        str(int(cfg.data.coyo.img2dataset.thread_count)),
        "--image_size",
        str(int(cfg.data.coyo.img2dataset.image_size)),
        "--resize_only_if_bigger",
        "True" if bool(cfg.data.coyo.img2dataset.resize_only_if_bigger) else "False",
        "--resize_mode",
        str(cfg.data.coyo.img2dataset.resize_mode),
        "--skip_reencode",
        "True" if bool(cfg.data.coyo.img2dataset.skip_reencode) else "False",
        "--save_additional_columns",
        add_cols_json,
        "--enable_wandb",
        "True" if bool(cfg.data.coyo.img2dataset.enable_wandb) else "False",
    ]
    _run(cmd)


def ensure_wds_shards(cfg: DictConfig) -> None:
    """
    Ensures that cfg.data.coyo.processed_dir has shards 00000.tar..000{num_shards-1}.tar
    Strategy:
      - if already present -> ok
      - try dvc pull (optional)
      - else build from open sources (HF parquet + img2dataset)
    """
    if not bool(cfg.data.coyo.enabled):
        raise RuntimeError(
            "cfg.data.coyo.enabled=false, but WDS shards are required to build final dataset."
        )

    wds_dir = _resolve(cfg.data.coyo.processed_dir)
    num_shards = int(cfg.data.coyo.num_shards)
    samples_per_shard = int(cfg.data.coyo.samples_per_shard)
    max_samples = num_shards * samples_per_shard

    if _have_required_shards(wds_dir, num_shards):
        return

    # Build from open sources
    parquet_files = download_coyo_parquets(cfg)

    subset_parquet = (
        _resolve(cfg.data.coyo.raw_dir).parent
        / "coyo-700m_subset"
        / "part_subset.parquet"
    )
    columns = ["url", "text"] + list(cfg.data.coyo.img2dataset.additional_columns)

    make_subset_parquet(
        parquet_files=parquet_files,
        subset_path=subset_parquet,
        max_samples=max_samples,
        columns=columns,
    )

    run_img2dataset(cfg, subset_parquet=subset_parquet)

    if not _have_required_shards(wds_dir, num_shards):
        raise RuntimeError("Failed to create required WDS shards.")


def _file_nonempty(p: Path) -> bool:
    return p.exists() and p.is_file() and p.stat().st_size > 0


# Final build: WDS -> coyo_original (images + json) -> filter bad samples
def build_coyo_original_from_wds(cfg: DictConfig) -> None:
    """
    Runs:
      tools/coyo_wds_to_original.py
      tools/jsonl_to_json_array.py
      tools/check_broken_images.py
    """
    # Resolve paths from cfg.data.build
    wds_dir = _resolve(cfg.data.build.wds_dir)
    out_images = _resolve(cfg.data.build.out_images)
    out_jsonl = _resolve(cfg.data.build.out_jsonl)
    out_json = _resolve(cfg.data.build.out_json)
    out_json_filtered = _resolve(cfg.data.build.out_json_filtered)
    bad_report = _resolve(cfg.data.build.bad_report)
    min_size = int(cfg.data.build.min_size)
    max_aspect = float(cfg.data.build.max_aspect)

    out_images.mkdir(parents=True, exist_ok=True)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json_filtered.parent.mkdir(parents=True, exist_ok=True)
    bad_report.parent.mkdir(parents=True, exist_ok=True)

    images_ready = _dir_nonempty(out_images)

    if not images_ready:
        # Remove stale outputs (safe)
        for p in [out_jsonl, out_json, out_json_filtered]:
            if p.exists():
                p.unlink()

        # 1) WDS -> images + jsonl
        script_coyo = _find_tool_script("coyo_wds_to_original.py")
        _run(
            [
                sys.executable,
                str(script_coyo),
                "--wds_dir",
                str(wds_dir),
                "--out_images",
                str(out_images),
                "--out_jsonl",
                str(out_jsonl),
                "--max_samples",
                "0",
            ]
        )
        images_ready = _dir_nonempty(out_images)

        if not images_ready or not _file_nonempty(out_jsonl):
            raise RuntimeError(
                "WDS -> coyo_original step finished, but outputs look empty. "
                f"images_ready={images_ready}, jsonl_exists={out_jsonl.exists()}."
            )
    else:
        if not _file_nonempty(out_jsonl):
            raise RuntimeError(
                "Found non-empty out_images, but out_jsonl is missing/empty. "
            )

    # 2) jsonl -> json array
    if not _file_nonempty(out_json):
        script_jsonl = _find_tool_script("jsonl_to_json_array.py")
        _run(
            [
                sys.executable,
                str(script_jsonl),
                "--in_jsonl",
                str(out_jsonl),
                "--out_json",
                str(out_json),
            ]
        )
        if not _file_nonempty(out_json):
            raise RuntimeError(
                "jsonl_to_json_array finished, but out_json is missing/empty."
            )

    # 3) filter broken samples -> data_filtered.json (+ report)
    if not _file_nonempty(out_json_filtered):
        script_check = _find_tool_script("check_broken_images.py")
        _run(
            [
                sys.executable,
                str(script_check),
                "--in_json",
                str(out_json),
                "--image_root",
                str(out_images),
                "--out_json",
                str(out_json_filtered),
                "--bad_report",
                str(bad_report),
                "--min_size",
                str(min_size),
                "--max_aspect",
                str(max_aspect),
            ]
        )
        if not _file_nonempty(out_json_filtered):
            raise RuntimeError(
                "check_data finished, but out_json_filtered is missing/empty."
            )


def download_data(cfg: DictConfig) -> None:
    """
    REQUIRED BY ASSIGNMENT (local storage case).

    Builds the final training dataset from open sources if it's not available:
      - ensure WDS shards exist (HF parquet -> img2dataset)
      - convert to coyo_original
      - filter broken samples
    """
    ensure_wds_shards(cfg)
    build_coyo_original_from_wds(cfg)


def ensure_training_data(cfg: DictConfig) -> None:
    """
    Main entry point for train.py / infer.py.

    1) If final dataset (json_file + root_path) already exists -> OK
    2) Else try `dvc pull` for final_root (best-effort)
    3) Else build from open sources via download_data()
    """
    if not bool(cfg.data.build.enabled):
        # If build is disabled, we only check existence.
        if not _final_dataset_ready(cfg):
            raise RuntimeError(
                "Training data not found, and cfg.data.build.enabled=false. "
                "Either enable build or provide data/processed/coyo_original."
            )
        return

    # 1) Already present
    if _final_dataset_ready(cfg):
        return

    # 2) Try DVC pull for final_root (best-effort)
    if bool(cfg.data.build.dvc.try_pull_first):
        final_root = _resolve(cfg.data.build.final_root)
        remote = str(cfg.data.build.dvc.remote_name)
        jobs = int(cfg.data.build.dvc.jobs)
        ok = _try_dvc_pull(final_root, remote=remote, jobs=jobs)
        if ok and _final_dataset_ready(cfg):
            return

    # 3) Fallback: build from open sources
    download_data(cfg)

    if not _final_dataset_ready(cfg):
        raise RuntimeError("Failed to prepare final training dataset (coyo_original).")
