#!/usr/bin/env python3
# jsonl_to_hf_llava_mix.py
#
# Converts your JSONL (image + conversations) into a HF dataset matching
# trl-lib/llava-instruct-mix style:
#   - images:     list[Image] (usually len=1)
#   - prompt:     list[{role, content}]  (all turns except last assistant)
#   - completion: list[{role, content}]  (last assistant turn)
#
# Writes Parquet per split
#
# Example:
# python src/jsonl_to_hf.py \
#     --train_jsonl data/highlighted_images_v4.5/train_output.jsonl \
#     --val_jsonl data/highlighted_images_v4.5/val_output.jsonl \
#     --test_jsonl data/highlighted_images_v4.5/test_output.jsonl \
#     --image_base_dir data/highlighted_images_v4.5 \
#     --out_dir data/highlighted_images_v4.5_hf \
#     --mode embed \
#     --strip_image_token

import argparse
import json
import os
import re
import shutil
from typing import Any, Dict, Iterator, List as PyList, Optional, Tuple

from datasets import Dataset, DatasetDict, Features, Sequence, Value, Image
from datasets.features import List as HFList


IMG_TOKEN_RE = re.compile(r"^\s*<image>\s*\n?", re.IGNORECASE)

FROM_TO_ROLE = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
}


def info(msg: str) -> None:
    print(f"[INFO] {msg}")


def warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def read_jsonl_iter(path: str) -> Iterator[Tuple[int, Dict[str, Any]]]:
    """Yield (line_number, obj)."""
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Bad JSON in {path}:{ln}: {e}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"Expected JSON object in {path}:{ln}, got {type(obj)}")
            yield ln, obj


def resolve_image_path(image_field: str, image_base_dir: Optional[str]) -> str:
    if os.path.isabs(image_field):
        return image_field
    if not image_base_dir:
        return image_field
    return os.path.join(image_base_dir, image_field)


def abspath_from_root(path: Optional[str], root_dir: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    if os.path.isabs(path):
        return path
    if root_dir:
        return os.path.abspath(os.path.join(root_dir, path))
    return os.path.abspath(path)


def normalize_messages(conversations: Any, strip_image_token: bool) -> Optional[list]:
    if not isinstance(conversations, list) or len(conversations) < 2:
        return None

    msgs = []
    for t in conversations:
        if not isinstance(t, dict):
            continue
        frm = str(t.get("from", "")).strip()
        role = FROM_TO_ROLE.get(frm)
        if role is None:
            continue
        content = str(t.get("value", ""))
        if strip_image_token and role == "user":
            content = IMG_TOKEN_RE.sub("", content)
        msgs.append({"role": role, "content": content})

    if len(msgs) < 2:
        return None
    if msgs[-1]["role"] != "assistant":
        return None
    return msgs


def pack_turns(turns: PyList[Dict[str, Any]]) -> PyList[Dict[str, str]]:
    """Normalize turns into list of {role, content} dicts."""
    packed = []
    for t in turns:
        packed.append(
            {
                "role": str(t.get("role", "")).strip(),
                "content": str(t.get("content", "")).strip(),
            }
        )
    return packed


def image_item_from_path(
    src_path: str,
    out_dir: str,
    mode: str,
    make_abs_paths: bool,
    rel_hint: Optional[str] = None,
    fs_out_dir: Optional[str] = None,
    root_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Return a dict compatible with datasets.Image() encoding:
      {"path": "..."}  or  {"path": "...", "bytes": b"..."}
    IMPORTANT: This must return a dict, never a list.
    """
    if mode == "link":
        if make_abs_paths and not os.path.isabs(src_path):
            base = root_dir or os.getcwd()
            p = os.path.abspath(os.path.join(base, src_path))
        else:
            p = src_path
        return {"path": p}

    if mode == "copy":
        copy_base = fs_out_dir
        if copy_base is None:
            if os.path.isabs(out_dir):
                copy_base = out_dir
            else:
                base = root_dir or os.getcwd()
                copy_base = os.path.abspath(os.path.join(base, out_dir))

        images_dir = os.path.join(copy_base, "images")
        os.makedirs(images_dir, exist_ok=True)

        # Preserve relative structure from JSONL image field to avoid filename collisions
        rel = (rel_hint or os.path.basename(src_path)).lstrip("./")
        dst = os.path.join(images_dir, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)

        if not os.path.exists(dst):
            shutil.copy2(src_path, dst)

        stored_rel = os.path.join("images", rel)
        stored = os.path.join(out_dir, stored_rel)
        if make_abs_paths:
            if os.path.isabs(stored):
                stored = stored
            else:
                base = root_dir or os.getcwd()
                stored = os.path.abspath(os.path.join(base, stored))
        return {"path": stored}

    if mode == "embed":
        with open(src_path, "rb") as f:
            b = f.read()
        return {"path": os.path.basename(src_path), "bytes": b}

    raise ValueError(f"Unknown mode: {mode}")


def validate_row(row: Dict[str, Any], jsonl_path: str, ln: int) -> None:
    # images must be list[dict], each dict for Image feature
    imgs = row.get("images")
    if not isinstance(imgs, list) or len(imgs) == 0 or not isinstance(imgs[0], dict):
        raise TypeError(
            f"{jsonl_path}:{ln} images must be list[dict], got {type(imgs)} "
            f"first={type(imgs[0]) if isinstance(imgs, list) and imgs else None}"
        )

    # prompt must be list[dict]
    prompt = row.get("prompt")
    if not isinstance(prompt, list) or any(not isinstance(x, dict) for x in prompt):
        raise TypeError(f"{jsonl_path}:{ln} prompt must be list[dict], got type={type(prompt)}")
    if any(not isinstance(x.get("role"), str) or not isinstance(x.get("content"), str) for x in prompt):
        raise TypeError(f"{jsonl_path}:{ln} prompt items must have string role/content")

    # completion must be list[dict] of length 1
    completion = row.get("completion")
    if not isinstance(completion, list) or len(completion) != 1 or not isinstance(completion[0], dict):
        raise TypeError(
            f"{jsonl_path}:{ln} completion must be list[dict] length=1, got type={type(completion)}"
        )
    if any(
        not isinstance(x.get("role"), str) or not isinstance(x.get("content"), str)
        for x in completion
    ):
        raise TypeError(f"{jsonl_path}:{ln} completion items must have string role/content")


def iter_hf_rows(
    jsonl_path: str,
    image_base_dir: Optional[str],
    out_dir: str,
    mode: str,
    strip_image_token: bool,
    make_abs_paths: bool,
    max_skips_logged: int = 30,
    root_dir: Optional[str] = None,
) -> Iterator[Dict[str, Any]]:
    total = 0
    kept = 0
    skipped = 0

    root_dir = os.path.abspath(root_dir) if root_dir else None
    fs_out_dir = abspath_from_root(out_dir, root_dir)
    fs_image_base_dir = abspath_from_root(image_base_dir, root_dir) if image_base_dir else root_dir

    for ln, ex in read_jsonl_iter(jsonl_path):
        total += 1

        image_field = ex.get("image")
        if not image_field:
            skipped += 1
            if skipped <= max_skips_logged:
                warn(f"{jsonl_path}:{ln} missing 'image', skipping")
            continue

        src_path_fs = resolve_image_path(str(image_field), fs_image_base_dir)
        if mode in ("link", "copy", "embed") and not os.path.isfile(src_path_fs):
            skipped += 1
            if skipped <= max_skips_logged:
                warn(f"{jsonl_path}:{ln} image not found: {src_path_fs}, skipping")
            continue

        msgs = normalize_messages(ex.get("conversations"), strip_image_token=strip_image_token)
        if msgs is None:
            skipped += 1
            if skipped <= max_skips_logged:
                warn(f"{jsonl_path}:{ln} bad conversations or last turn not assistant, skipping")
            continue

        prompt = pack_turns(msgs[:-1])
        completion = pack_turns([msgs[-1]])

        if mode == "link":
            src_path_store = (
                src_path_fs
                if make_abs_paths
                else resolve_image_path(str(image_field), image_base_dir)
            )
            item_src_path = src_path_store
        else:
            item_src_path = src_path_fs

        try:
            img_item = image_item_from_path(
                src_path=item_src_path,
                out_dir=out_dir,
                mode=mode,
                make_abs_paths=make_abs_paths,
                rel_hint=str(image_field),  # key for copy mode collision-free
                fs_out_dir=fs_out_dir,
                root_dir=root_dir,
            )
        except FileNotFoundError:
            skipped += 1
            if skipped <= max_skips_logged:
                warn(f"{jsonl_path}:{ln} image not found during load: {src_path_fs}, skipping")
            continue

        # CRITICAL: wrap exactly once
        row = {
            "images": [img_item],
            "prompt": prompt,
            "completion": completion,
        }

        # fail fast with a helpful message if something is wrong
        validate_row(row, jsonl_path, ln)

        kept += 1
        yield row

    info(f"{jsonl_path}: total={total:,} kept={kept:,} skipped={skipped:,}")


def build_dataset_from_jsonl(
    jsonl_path: str,
    image_base_dir: Optional[str],
    out_dir: str,
    mode: str,
    strip_image_token: bool,
    make_abs_paths: bool,
    writer_batch_size: int,
    root_dir: Optional[str],
) -> Dataset:
    features = Features(
        {
            "images": Sequence(Image()),
            "prompt": HFList({"role": Value("string"), "content": Value("string")}),
            "completion": HFList({"role": Value("string"), "content": Value("string")}),
        }
    )

    return Dataset.from_generator(
        iter_hf_rows,
        features=features,
        writer_batch_size=writer_batch_size,
        gen_kwargs={
            "jsonl_path": jsonl_path,
            "image_base_dir": image_base_dir,
            "out_dir": out_dir,
            "mode": mode,
            "strip_image_token": strip_image_token,
            "make_abs_paths": make_abs_paths,
            "root_dir": root_dir,
        },
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--val_jsonl", required=True)
    ap.add_argument("--test_jsonl", required=True)

    ap.add_argument("--image_base_dir", default=None)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--mode", choices=["link", "copy", "embed"], default="link")
    ap.add_argument("--strip_image_token", action="store_true")
    ap.add_argument("--make_abs_paths", action="store_true")
    ap.add_argument("--writer_batch_size", type=int, default=1000)

    ap.add_argument("--save_to_disk", action="store_true")

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    root_dir = os.path.abspath(os.getcwd())

    splits = {
        "train": args.train_jsonl,
        "val": args.val_jsonl,
        "test": args.test_jsonl,
    }

    ds_dict = {}
    for split, path in splits.items():
        info(f"Building split={split} from {path}")
        ds = build_dataset_from_jsonl(
            jsonl_path=path,
            image_base_dir=args.image_base_dir,
            out_dir=args.out_dir,
            mode=args.mode,
            strip_image_token=args.strip_image_token,
            make_abs_paths=args.make_abs_paths,
            writer_batch_size=args.writer_batch_size,
            root_dir=root_dir,
        )

        out_parquet = os.path.join(args.out_dir, f"{split}.parquet")
        try:
            ds.to_parquet(out_parquet)
        except FileNotFoundError as e:
            warn(
                f"{split} parquet size estimate failed due to missing image: {e}. "
                f"Retrying with batch_size={args.writer_batch_size}."
            )
            ds.to_parquet(out_parquet, batch_size=args.writer_batch_size)
        info(f"Wrote {len(ds):,} rows to {out_parquet}")

        if len(ds) > 0:
            try:
                ex0 = ds[0]
            except (FileNotFoundError, OSError) as e:
                warn(f"{split}[0] sample decode failed: {e}")
            else:
                info(f"{split}[0] images={ex0['images']}")
                info(
                    f"{split}[0] prompt_turns={len(ex0['prompt'])} completion_turns={len(ex0['completion'])}"
                )

        ds_dict[split] = ds

    dsd = DatasetDict(ds_dict)

    if args.save_to_disk:
        out_disk = os.path.join(args.out_dir, "hf_dataset")
        dsd.save_to_disk(out_disk)
        info(f"Saved DatasetDict to {out_disk}")


if __name__ == "__main__":
    main()
