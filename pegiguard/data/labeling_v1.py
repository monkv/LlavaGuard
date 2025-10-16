import csv
import os
import argparse
import glob
import base64
from io import BytesIO
from PIL import Image
from datasets import load_dataset
from openai import OpenAI

import sys
sys.path.append('/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/am84fuxo/LlavaGuard')
from llavaguard_config import local_image_dirs, local_data_dir
from llavaguard.taxonomy.PEGI.PEGI_Graph import policy_graph
from pegiguard.data.prompts_v2 import build_prompt

MODEL_ID = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
CHECK_POINT = 100

def key_of(entry: dict) -> tuple:
    """Create a unique key for an entry based on image path and category."""
    return (entry.get("image_path",""), entry.get("category",""))

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
        #PNG preserves transparency
        img.save(bio, format="PNG")
        mime = "image/png"
    else:
        #Normalize for JPEG
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
        flat_rows.append({
            "image_path": r.get("image_path"),
            "subcategory": r.get("subcategory"),
            "rating": r.get("rating"),
            "assessment": r.get("assessment"),
            "review": r.get("review", [])
        })

    fieldnames = ["image_path", "subcategory", "rating", "assessment", "review"]
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            w.writeheader()
        w.writerows(flat_rows)


def process_all_images(images_root, output_file, system_rule, batch_size=20):
    """Process all images in a directory, assessing them with the model and saving results."""
    #create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    file_exists = os.path.isfile(output_file)
    saved = []
    csv_file = open(output_file, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=["image_path", "subcategory", "rating", "assessment", "review"])

    if not file_exists:
        writer.writeheader()  #write header only once
    existing = set()
    buffer  = []

    ext = ("*.jpg")
    roots = [os.fspath(v) for v in images_root.values()]
    image_paths: list[str] = []
    for root in roots:
        image_paths.extend(glob.glob(os.path.join(root, "**", ext), recursive=True))
    image_paths.sort()

    processed = 0
    total_images = len(image_paths)

    client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

    for idx, img_path in enumerate(image_paths, start=1):
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Skipping unreadable image: {img_path} ({e})")
            continue

        for category, category_details in policy_graph.items():
            for subcategory, graph_info in category_details.items():
                if subcategory == "46. Child Endangerment" or subcategory == "47. Child Sexual Abuse":
                    continue
                prompt = build_prompt(category, subcategory, graph_info)

                image = Image.open(img_path).convert("RGB")
                data_url = pil_to_data_url(image)
                #print(prompt)
                messages = [
                    {"role": "system", "content": system_rule},
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ]},
                ]

                resp = client.chat.completions.create(
                    model=MODEL_ID,
                    messages=messages,
                    max_tokens=3,
                )
                response = resp.choices[0].message.content.strip()

                if not (response == "NO" or response == "YES"):
                    print(f"Unknown answer - {response}, img - {img_path}")
                    continue
                record = {
                    "image": img_path,
                    "subcategory": subcategory,
                    "rating": graph_info["rating"],
                    "assessment": response,
                    "review": "",
                }
                buffer.append(record)
                k = key_of(record)
                if k in existing:
                    continue
                existing.add(k)
                buffer.append(record)

        processed += 1
        if processed % batch_size == 0:
            processed += 1
            writer.writerows(buffer)
            csv_file.flush()
            buffer.clear()
            print(f"Progress: {processed}/{total_images} images processed.")

    writer.writerows(buffer)
    csv_file.flush()
    buffer.clear()
    print(f"Progress: {processed}/{total_images} images processed.")
    print(f"Done. Processed {processed} images. Output in {output_file}")

def process_hf_dataset(output_file: str, system_rule: str, batch_size: int = 20):
    """
    Relabel images from a HuggingFace dataset - yiting/UnsafeBench.
    Writes the same CSV schema as your directory pipeline.
    """
    data_dir = "/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/am84fuxo/data/UnsafeBench"
    ds = load_dataset("yiting/UnsafeBench", cache_dir=data_dir)["train"]

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    file_exists = os.path.isfile(output_file)
    csv_file = open(output_file, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=["idx", "image_path", "subcategory", "rating", "assessment", "review"])
    if not file_exists:
        writer.writeheader()

    buffer = []
    processed = 0
    total_images = len(ds)
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

    for idx, sample in enumerate(ds):
        image = sample["image"]
        try:
            data_url = pil_to_data_url(image)
        except Exception as e:
            print(f"[Skipping img: {idx} due to encode fail ({e})")
            continue

        for category, category_details in policy_graph.items():
            for subcategory, graph_info in category_details.items():
                if subcategory == "46. Child Endangerment" or subcategory == "47. Child Sexual Abuse":
                    continue
                prompt = build_prompt(category, subcategory, graph_info)
                messages = [
                    {"role": "system", "content": system_rule},
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ]},
                ]

                resp = client.chat.completions.create(
                    model=MODEL_ID,
                    messages=messages,
                    max_tokens=3,
                )
                response = resp.choices[0].message.content.strip()

                if not (response == "NO" or response == "YES"):
                    print(f"Unknown answer - {response}, img - {idx}")
                    continue

                record = {
                    "idx": idx,
                    "image_path": image,
                    "subcategory": subcategory,
                    "rating": graph_info.get("rating"),
                    "assessment": response,
                    "review": "",
                }
                buffer.append(record)

        processed += 1
        if processed % batch_size == 0:
            processed += 1
            writer.writerows(buffer)
            csv_file.flush()
            buffer.clear()
            print(f"Progress: {processed}/{total_images} images processed.")
    writer.writerows(buffer)
    csv_file.flush()
    buffer.clear()
    print(f"Progress: {processed}/{total_images} images processed.")
    print(f"Done. Processed {processed} images. Output in {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_root", default=local_image_dirs)
    parser.add_argument("--output_file", default=f"{local_data_dir}/labels/pegi/v3/labels_all.csv")
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--model_id", default="meta-llama/Llama-4-Scout-17B-16E-Instruct")
    parser.add_argument("--offload_dir", default="/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/am84fuxo/offload_cache")
    parser.add_argument("--mode", choices=["dir", "hf"], default="dir")
    args = parser.parse_args()

    SYSTEM_RULE = (
        "You are a careful, literal safety reviewer. "
        "Output ONLY a single YES or NO."
        "No extra text, no markdown, no explanations outside of the decision."
    )

    if args.mode == "dir":
        process_all_images(
            images_root=args.images_root,
            output_file=args.output_file,
            system_rule=SYSTEM_RULE,
            batch_size=args.batch_size,
        )
    else:
        process_hf_dataset(
            output_file=args.output_file,
            system_rule=SYSTEM_RULE,
            batch_size=args.batch_size,
        )
