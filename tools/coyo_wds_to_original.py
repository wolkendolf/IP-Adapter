from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path

IMG_EXTS = {"jpg", "jpeg", "png", "webp"}


def iter_tar_members(tf: tarfile.TarFile):
    for m in tf:
        if m.isfile():
            yield m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wds_dir", required=True)
    ap.add_argument("--out_images", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--max_samples", type=int, default=0, help="0 = no limit")
    args = ap.parse_args()

    wds_dir = Path(args.wds_dir)
    out_images = Path(args.out_images)
    out_images.mkdir(parents=True, exist_ok=True)
    out_jsonl = Path(args.out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    written = 0

    with open(out_jsonl, "w", encoding="utf-8") as f_out:
        for tar_path in sorted(wds_dir.glob("*.tar")):
            print("Processing", tar_path.name)
            with tarfile.open(tar_path, "r:*") as tf:
                # собираем по ключу: key -> {"img": bytes, "txt": str}
                cur_key = None
                buf = {}

                def flush():
                    nonlocal written
                    if not cur_key:
                        return
                    img = buf.get("img_bytes")
                    ext = buf.get("img_ext")
                    txt = buf.get("txt")
                    if img is None or ext is None or not txt:
                        return

                    # сохраняем картинку
                    img_name = f"{cur_key}.{ext}"
                    img_path = out_images / img_name
                    if not img_path.exists():
                        img_path.write_bytes(img)

                    # запись в jsonl: image_file относительный к out_images
                    rec = {"image_file": img_name, "text": txt}
                    f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    written += 1

                for m in iter_tar_members(tf):
                    name = Path(m.name).name
                    if "." not in name:
                        continue
                    key, ext = name.rsplit(".", 1)
                    ext = ext.lower()

                    if cur_key is None:
                        cur_key = key
                    if key != cur_key:
                        flush()
                        buf = {}
                        cur_key = key

                    f = tf.extractfile(m)
                    if f is None:
                        continue
                    data = f.read()

                    if ext in IMG_EXTS:
                        buf["img_bytes"] = data
                        buf["img_ext"] = ext
                    elif ext == "txt":
                        buf["txt"] = data.decode("utf-8", errors="replace").strip()

                    if args.max_samples and written >= args.max_samples:
                        break

                flush()

            if args.max_samples and written >= args.max_samples:
                break

    print("Wrote samples:", written)
    print("JSONL:", out_jsonl)
    print("Images:", out_images)


if __name__ == "__main__":
    main()
