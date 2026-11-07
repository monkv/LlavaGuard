"""Label image subcategories concurrently and resume incomplete output files."""

from __future__ import annotations

import argparse
import base64
import csv
import glob
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from typing import Optional

import pandas as pd
from PIL import Image


sys.path.append('/pfss/mlde/workspaces/mlde_wsp_PI_Kersting/am84fuxo/LlavaGuard')

from llavaguard_config import local_image_dirs, local_data_dir
from llavaguard.taxonomy.PEGI.PEGI_Graph import policy_graph
from pegiguard.data.prompts import build_prompt

MODEL_ID = {
    "LLAMA_4": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "QWEN_3.5": "Qwen/Qwen3.5-35B-A3B",
    "GEMMA_4b": "google/gemma-3-4b-it",
    "GEMMA_12b": "google/gemma-3-12b-it"
}
CHECK_POINT = 100
SKIP_SUBCATEGORIES = {"46. Child Endangerment", "47. Child Sexual Abuse"}
_thread_local = threading.local()
unknown_lock = threading.Lock()


def key_of(entry: dict) -> tuple:
    """Create a unique key for an entry based on image path and category."""
    return (entry.get("image_path", ""), entry.get("category", ""))


def fix_image_path(img_path: str) -> str:
    return img_path.replace(
        "/mlde_wsp_KIServiceCenter/",
        "/mlde_wsp_PI_Kersting/",
    )


def pil_to_data_url(img: Image.Image) -> str:
    """
    Converts PIL image to a data URL.
    - If the image has alpha (RGBA/LA/P), encode as PNG to preserve transparency.
    - Otherwise, encode as JPEG after converting to RGB.
    """
    mode = img.mode
    has_alpha = ("A" in mode) or (mode == "P")
    bio = BytesIO()
    if has_alpha:
        img.save(bio, format="PNG")
        mime = "image/png"
    else:
        if mode not in ("RGB",):
            img = img.convert("RGB")
        img.save(bio, format="JPEG")
        mime = "image/jpeg"
    b64 = base64.b64encode(bio.getvalue()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def append_rows_to_csv(rows, csv_path):
    """Append assessment records to a CSV, creating it with a header if needed."""
    flat_rows = []
    for r in rows:
        flat_rows.append(
            {
                "image_path": r.get("image_path"),
                "subcategory": r.get("subcategory"),
                "rating": r.get("rating"),
                "assessment": r.get("assessment"),
            }
        )

    fieldnames = ["image_path", "subcategory", "rating", "assessment"]
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            w.writeheader()
        w.writerows(flat_rows)


def get_client() -> OpenAI:
    client = getattr(_thread_local, "client", None)
    if client is None:
        from openai import OpenAI
        client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
        _thread_local.client = client
    return client


def iter_policy_items():
    for category, category_details in policy_graph.items():
        for subcategory, graph_info in category_details.items():
            if subcategory in SKIP_SUBCATEGORIES:
                continue
            yield category, subcategory, graph_info


def classify_single_policy(
    *,
    model_id: str,
    system_rule: str,
    data_url: str,
    category: str,
    subcategory: str,
    graph_info: dict,
) -> Optional[dict]:
    prompt = build_prompt(category, subcategory, graph_info)
    messages_to_model: list[dict] = [
        {"role": "system", "content": system_rule},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        },
    ]

    client = get_client()
    resp = client.chat.completions.create(
        model=model_id,
        messages=messages_to_model,
        max_tokens=3,
        temperature=0,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    content = resp.choices[0].message.content
    if content is None:
        return None

    content = content.strip().upper()
    if content not in {"YES", "NO"}:
        return {
            "subcategory": subcategory,
            "rating": graph_info.get("rating"),
            "assessment": "",
            "error": f"Unknown answer: {content}",
        }

    return {
        "subcategory": subcategory,
        "rating": graph_info.get("rating"),
        "assessment": content,
        "error": None,
    }


def classify_image_across_policies(
    *,
    model_id: str,
    system_rule: str,
    data_url: str,
    max_workers: int,
    skip_subcategories=None,
) -> list[dict]:
    skipped = set(skip_subcategories or ())
    tasks = [item for item in iter_policy_items() if item[1] not in skipped]
    results: list[dict] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                classify_single_policy,
                model_id=model_id,
                system_rule=system_rule,
                data_url=data_url,
                category=category,
                subcategory=subcategory,
                graph_info=graph_info,
            ): (subcategory, graph_info)
            for category, subcategory, graph_info in tasks
        }

        for future in as_completed(futures):
            subcategory, graph_info = futures[future]
            try:
                result = future.result()
            except Exception as e:
                results.append(
                    {
                        "subcategory": subcategory,
                        "rating": graph_info.get("rating"),
                        "assessment": None,
                        "error": str(e),
                    }
                )
                continue

            if result is None:
                results.append(
                    {
                        "subcategory": subcategory,
                        "rating": graph_info.get("rating"),
                        "assessment": None,
                        "error": "Empty response",
                    }
                )
            else:
                results.append(result)

    results.sort(key=lambda x: x["subcategory"])
    return results


def fill_missing_assessments(
    csv_file: str,
    system_rule: str,
    model_id: str,
    batch_size: int = 20,
    max_workers: int = 16,
):
    """Fill missing assessments in a CSV (same format as llavaguard.csv).

    Only re-runs the model for categories where the assessment is missing.
    """
    df = pd.read_csv(csv_file, dtype=str)
    if "assessment" not in df.columns:
        raise ValueError("Input CSV must contain an 'assessment' column")

    missing_mask = ~df["assessment"].fillna("").str.strip().str.upper().isin({"YES", "NO"})
    missing_mask &= ~df["subcategory"].isin(SKIP_SUBCATEGORIES)
    df["image_path"] = df["image_path"].map(lambda path: fix_image_path(path) if pd.notna(path) else path)
    if not missing_mask.any():
        print("No missing assessments found.")
        return

    # Build a quick lookup of {subcategory: (category, graph_info)}.
    subcat_map = {
        subcat: (cat, info)
        for cat, cat_info in policy_graph.items()
        for subcat, info in cat_info.items()
    }

    unknown_file = os.path.join(os.path.dirname(csv_file), "unknown_report_2.csv")
    unknown_exists = os.path.isfile(unknown_file)
    unknown_f = open(unknown_file, "a", newline="", encoding="utf-8")
    unknown_writer = csv.DictWriter(unknown_f, fieldnames=["image_path", "subcategory", "rating"])
    if not unknown_exists:
        unknown_writer.writeheader()

    missing_df = df.loc[missing_mask]
    unique_images = missing_df["image_path"].dropna().unique()

    processed = 0
    for img_path in unique_images:
        try:
            image = Image.open(img_path).convert("RGB")
            data_url = pil_to_data_url(image)
        except Exception as e:
            print(f"Skipping unreadable image: {img_path} ({e})")
            continue

        # Only classify the missing subcategories for this image.
        missing_subcats = missing_df[missing_df["image_path"] == img_path]["subcategory"].unique()
        tasks = []
        for subcat in missing_subcats:
            if subcat not in subcat_map or subcat in SKIP_SUBCATEGORIES:
                print(f"Skipping unknown subcategory: {subcat}")
                continue
            cat, graph_info = subcat_map[subcat]
            tasks.append((cat, subcat, graph_info))

        if not tasks:
            processed += 1
            continue

        # Use parallelism for the subcategories, since the model calls are independent.
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    classify_single_policy,
                    model_id=model_id,
                    system_rule=system_rule,
                    data_url=data_url,
                    category=cat,
                    subcategory=subcat,
                    graph_info=graph_info,
                ): subcat
                for cat, subcat, graph_info in tasks
            }

            results = {}
            for fut in as_completed(futures):
                subcat = futures[fut]
                try:
                    results[subcat] = fut.result()
                except Exception as e:
                    results[subcat] = {
                        "subcategory": subcat,
                        "rating": None,
                        "assessment": "",
                        "error": str(e),
                    }

        # Apply updates to the rows in the original dataframe.
        locs = (df["image_path"] == img_path) & missing_mask
        for idx in df[locs].index:
            subcat = df.at[idx, "subcategory"]
            res = results.get(subcat)
            if res is None:
                continue

            if res.get("assessment") not in {"YES", "NO"}:
                with unknown_lock:
                    unknown_writer.writerow(
                        {
                            "image_path": img_path,
                            "subcategory": subcat,
                            "rating": res.get("rating"),
                        }
                    )

            df.at[idx, "assessment"] = res.get("assessment", "") or ""

        processed += 1
        if processed % batch_size == 0:
            print(f"Filled assessments for {processed}/{len(unique_images)} images")
            # Write intermediate results to file in batches.
            df.to_csv(csv_file, index=False)

    unknown_f.close()

    # Final overwrite input file in-place.
    df.to_csv(csv_file, index=False)
    print(f"Done. Filled missing assessments for {processed} images (output: {csv_file})")


def process_all_images_from_csv(
    input_file,
    output_file,
    system_rule,
    batch_size,
    model_id,
    max_workers,
):
    """Process images listed in an existing CSV, skipping images already present in output_file."""
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    file_exists = os.path.isfile(output_file) and os.path.getsize(output_file) > 0
    writer_file = open(output_file, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(
        writer_file,
        fieldnames=["image_path", "subcategory", "rating", "assessment"],
    )

    if not file_exists:
        writer.writeheader()
        writer_file.flush()

    unknown_file = os.path.join(os.path.dirname(output_file), "unknown_report.csv")
    unknown_exists = os.path.isfile(unknown_file)
    unknown_f = open(unknown_file, "a", newline="", encoding="utf-8")
    unknown_writer = csv.DictWriter(unknown_f, fieldnames=["image_path", "subcategory", "rating"])
    if not unknown_exists:
        unknown_writer.writeheader()

    existing_pairs = set()
    if file_exists and os.path.getsize(output_file) > 0:
        previous = pd.read_csv(output_file, usecols=["image_path", "subcategory"])
        existing_pairs = set(zip(previous["image_path"].map(fix_image_path), previous["subcategory"]))
    expected_subcategories = {subcategory for _, subcategory, _ in iter_policy_items()}

    df = pd.read_csv(input_file)
    unique_paths = df["image_path"].dropna().astype(str).unique()
    total_images = len(unique_paths)

    buffer = []
    processed_now = 0
    skipped = 0

    for old_img_path in unique_paths:
        img_path = fix_image_path(old_img_path)

        saved_subcategories = {subcategory for path, subcategory in existing_pairs if path == img_path}
        if expected_subcategories <= saved_subcategories:
            skipped += 1
            continue

        try:
            image = Image.open(img_path).convert("RGB")
            data_url = pil_to_data_url(image)
        except Exception as e:
            print(f"Skipping unreadable image: {img_path} ({e})")
            continue

        results = classify_image_across_policies(
            model_id=model_id,
            system_rule=system_rule,
            data_url=data_url,
            max_workers=max_workers,
            skip_subcategories=saved_subcategories,
        )

        for result in results:
            if result["assessment"] not in {"YES", "NO"}:
                print(f"Unknown answer - {result['error']}, img - {img_path}, subcategory - {result['subcategory']}")
                with unknown_lock:
                    unknown_writer.writerow({
                        "image_path": img_path,
                        "subcategory": result["subcategory"],
                        "rating": result["rating"],
                    })

            buffer.append(
                {
                    "image_path": img_path,
                    "subcategory": result["subcategory"],
                    "rating": result["rating"],
                    "assessment": result["assessment"],
                }
            )

        processed_now += 1
        existing_pairs.update((img_path, result["subcategory"]) for result in results)

        if processed_now % batch_size == 0:
            writer.writerows(buffer)
            writer_file.flush()
            buffer.clear()
            print(
                f"Progress: processed_now={processed_now}, skipped={skipped}, total_input={total_images}"
            )

    if buffer:
        writer.writerows(buffer)
        writer_file.flush()

    writer_file.close()
    unknown_f.close()
    print(
        f"Done. Newly processed: {processed_now}, skipped existing: {skipped}, total input images: {total_images}"
    )


def process_all_images(
    images_root,
    output_file,
    system_rule,
    model_id,
    batch_size=20,
    max_workers=16,
):
    """Process all images in a directory, assessing them with the model and saving results."""
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    file_exists = os.path.isfile(output_file) and os.path.getsize(output_file) > 0
    csv_file = open(output_file, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(
        csv_file,
        fieldnames=["image_path", "subcategory", "rating", "assessment"],
    )

    if not file_exists:
        writer.writeheader()

    unknown_file = os.path.join(os.path.dirname(output_file), "unknown_report.csv")
    unknown_exists = os.path.isfile(unknown_file)
    unknown_f = open(unknown_file, "a", newline="", encoding="utf-8")
    unknown_writer = csv.DictWriter(unknown_f, fieldnames=["image_path", "subcategory", "rating"])
    if not unknown_exists:
        unknown_writer.writeheader()

    buffer = []
    extensions = {".jpg", ".jpeg", ".png", ".webp"}
    roots = [os.fspath(v) for v in images_root.values()] if isinstance(images_root, dict) else [os.fspath(images_root)]
    image_paths: list[str] = []
    for root in roots:
        image_paths.extend(path for path in glob.glob(os.path.join(root, "**", "*"), recursive=True)
                           if os.path.isfile(path) and os.path.splitext(path)[1].lower() in extensions)
    image_paths = sorted(set(image_paths))

    existing_pairs = set()
    if file_exists:
        previous = pd.read_csv(output_file, usecols=["image_path", "subcategory"])
        existing_pairs = set(zip(previous["image_path"].map(fix_image_path), previous["subcategory"]))
    expected_subcategories = {subcategory for _, subcategory, _ in iter_policy_items()}
    processed = 0
    total_images = len(image_paths)

    for img_path in image_paths:
        saved_subcategories = {subcategory for path, subcategory in existing_pairs if path == img_path}
        if expected_subcategories <= saved_subcategories:
            continue
        try:
            image = Image.open(img_path).convert("RGB")
            data_url = pil_to_data_url(image)
        except Exception as e:
            print(f"Skipping unreadable image: {img_path} ({e})")
            continue

        results = classify_image_across_policies(
            model_id=model_id,
            system_rule=system_rule,
            data_url=data_url,
            max_workers=max_workers,
            skip_subcategories=saved_subcategories,
        )

        for result in results:
            if result["assessment"] not in {"YES", "NO"}:
                print(f"Unknown answer - {result['error']}, img - {img_path}, subcategory - {result['subcategory']}")
                with unknown_lock:
                    unknown_writer.writerow({
                        "image_path": img_path,
                        "subcategory": result["subcategory"],
                        "rating": result["rating"],
                    })

            buffer.append(
                {
                    "image_path": img_path,
                    "subcategory": result["subcategory"],
                    "rating": result["rating"],
                    "assessment": result["assessment"],
                }
            )

        processed += 1
        if processed % batch_size == 0:
            writer.writerows(buffer)
            csv_file.flush()
            buffer.clear()
            print(f"Progress: {processed}/{total_images} images processed.")

    if buffer:
        writer.writerows(buffer)
        csv_file.flush()
        buffer.clear()

    print(f"Progress: {processed}/{total_images} images processed.")
    print(f"Done. Processed {processed} images. Output in {output_file}")
    csv_file.close()
    unknown_f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_root", default=local_image_dirs)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--max_workers", type=int, default=16)
    parser.add_argument("--model_id", default=MODEL_ID["QWEN_3.5"])
    parser.add_argument(
        "--input_file",


        default=f"{local_data_dir}/labels/llama4_v1/unsafebench/unsafebench.csv",
    )
    parser.add_argument(
        "--output_file",
        default=f"{local_data_dir}/labels/qwen3_5/unsafebench/unsafebench.csv",
    )
    parser.add_argument("--mode", choices=["dir", "csv", "fill_missing"], default="fill_missing")
    args = parser.parse_args()

    SYSTEM_RULE = (
        "Output ONLY a single YES or NO."
        "No extra text, no markdown, no explanations outside of the decision."
    )

    print("Starting labeling with the following configuration:")
    print(
        f"Mode: {args.mode}",
        f"Model ID: {args.model_id}",
        f"Input File: {args.input_file}",
        f"Output File: {args.output_file}",
        f"Batch Size: {args.batch_size}",
        f"Max Workers: {args.max_workers}",
        sep="\n",
    )

    if args.mode == "dir":
        process_all_images(
            images_root=args.images_root,
            output_file=args.output_file,
            system_rule=SYSTEM_RULE,
            batch_size=args.batch_size,
            model_id=args.model_id,
            max_workers=args.max_workers,
        )
    elif args.mode == "csv":
        process_all_images_from_csv(
            input_file=args.input_file,
            output_file=args.output_file,
            system_rule=SYSTEM_RULE,
            batch_size=args.batch_size,
            model_id=args.model_id,
            max_workers=args.max_workers,
        )
    elif args.mode == "fill_missing":
        fill_missing_assessments(
            csv_file=args.output_file,
            system_rule=SYSTEM_RULE,
            model_id=args.model_id,
            batch_size=args.batch_size,
            max_workers=args.max_workers,
        )
