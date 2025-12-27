"""
Filter "bad" images out of an IP-Adapter JSON dataset.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image, UnidentifiedImageError


def check_image(
    img_path: Path,
    min_size: int,
    max_aspect: float,
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Returns: (is_ok, reason_if_bad, extra_info)
    """
    if not img_path.exists():
        return False, "missing_file", {"path": str(img_path)}

    try:
        # Avoid full decode; load just headers first.
        with Image.open(img_path) as im:
            w, h = im.size
            mode = im.mode

            if w <= 0 or h <= 0:
                return False, "invalid_dimensions", {"w": w, "h": h, "mode": mode}

            if min(w, h) < min_size:
                return False, "too_small", {"w": w, "h": h, "mode": mode}

            aspect = max(w, h) / max(1, min(w, h))
            if aspect > max_aspect:
                return (
                    False,
                    "aspect_too_large",
                    {"w": w, "h": h, "aspect": aspect, "mode": mode},
                )

            # Force a lightweight conversion check (catches some weird edge cases).
            # This may decode a bit, but helps prevent later failures.
            try:
                _ = im.convert("RGB")
            except Exception as e:  # noqa: BLE001
                return (
                    False,
                    "rgb_convert_failed",
                    {"w": w, "h": h, "mode": mode, "err": str(e)},
                )

            return True, "", {"w": w, "h": h, "mode": mode}

    except (UnidentifiedImageError, OSError) as e:
        return False, "cannot_open", {"path": str(img_path), "err": str(e)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_json", required=True, help="Path to data.json (JSON array)")
    ap.add_argument("--image_root", required=True, help="Root folder with images")
    ap.add_argument("--out_json", required=True, help="Output filtered JSON array")
    ap.add_argument(
        "--bad_report",
        default="",
        help="Optional JSONL path to write bad samples with reasons (empty disables)",
    )
    ap.add_argument(
        "--min_size",
        type=int,
        default=16,
        help="Minimum min(w,h) to keep (default: 16)",
    )
    ap.add_argument(
        "--max_aspect",
        type=float,
        default=20.0,
        help="Maximum aspect ratio max/min (default: 20.0)",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit of samples to scan (0 = all)",
    )
    args = ap.parse_args()

    in_json = Path(args.in_json)
    image_root = Path(args.image_root)
    out_json = Path(args.out_json)
    bad_report = Path(args.bad_report) if args.bad_report else None

    data: List[Dict[str, Any]] = json.loads(in_json.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{in_json} must be a JSON array of records")

    keep: List[Dict[str, Any]] = []
    bad: int = 0

    bad_fh = None
    if bad_report is not None:
        bad_report.parent.mkdir(parents=True, exist_ok=True)
        bad_fh = bad_report.open("w", encoding="utf-8")

    n = len(data) if args.limit <= 0 else min(len(data), args.limit)

    for i, rec in enumerate(data[:n], start=1):
        image_file = rec.get("image_file")
        if not image_file:
            bad += 1
            if bad_fh:
                bad_fh.write(
                    json.dumps(
                        {"reason": "missing_image_file_field", "record": rec},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            continue

        img_path = image_root / str(image_file)
        ok, reason, info = check_image(img_path, args.min_size, args.max_aspect)
        if ok:
            keep.append(rec)
        else:
            bad += 1
            if bad_fh:
                bad_fh.write(
                    json.dumps(
                        {
                            "reason": reason,
                            "info": info,
                            "record": rec,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        if i % 1000 == 0:
            print(f"scanned {i}/{n} | keep={len(keep)} bad={bad}", file=sys.stderr)

    if bad_fh:
        bad_fh.close()

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(keep, ensure_ascii=False), encoding="utf-8")

    print(f"Done.\nScanned: {n}\nKept: {len(keep)}\nBad: {bad}\nOutput: {out_json}")
    if bad_report is not None:
        print(f"Bad report: {bad_report}")


if __name__ == "__main__":
    main()
