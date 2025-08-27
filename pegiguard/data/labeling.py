import os
import argparse
import glob
import sys
sys.path.append('/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/am84fuxo/LlavaGuard')

import torch
from transformers import AutoProcessor, Llama4ForConditionalGeneration
from PIL import Image
from llavaguard_config import local_image_dirs, local_data_dir
from llavaguard.taxonomy.PEGI.PEGI_Graph import policy_graph

import pandas as pd
import json


MODEL_ID = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
CHECK_POINT = 100  

def build_prompt(category: str, subcategory: str, graph_info):
    edge = graph_info["edge"]
    examples = graph_info["examples"]
    rating = graph_info["rating"]
    
    prompt = f"""Top-level Category: {category}
        Subcategory to assess: {subcategory}

        Definition:
        - Edge: {edge}
        - PEGI rating: {rating}
        - Examples:  {examples}

        Assessment steps:
        - Default decision is "NO".
        - Return "YES" ONLY IF the image relates to the given safety category, meaning the image shows clear, visible evidence that matches at least ONE item in the example list above
        (literal match or close paraphrase).
        - Do NOT infer from mood, implication, text overlays, captions, or off-screen context.
        - If cues are uncertain/occluded/low-res/ambiguous, decide "NO".
        - If decision is "NO", set "rationale" to an empty string "".
        - If decision is "YES" explain in the "rationale" how the image relates to the given safety subcategory.

        Return EXACTLY ONE JSON object and NOTHING ELSE:
        {{
        "category": {subcategory},
        "decision": "YES" or "NO",
        "rating": {rating}
        "rationale": "string (leave empty if decision is NO)"
        }}
        """.strip()
    return prompt

def process_all_images(images_root, output_file, model, processor, system_rule, batch_size=20):
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                saved = json.load(f)
            if not isinstance(saved, list):
                saved = []
        except Exception:
            saved = []
    else:
        saved = []

    def key_of(entry: dict) -> tuple:
        return (entry.get("image_path",""), entry.get("category",""))

    existing = { key_of(e) for e in saved }
    buffer  = []

    exts = ("*.jpg")
    image_paths = []
    for ext in exts:
        image_paths.extend(glob.glob(os.path.join(images_root, "**", ext), recursive=True))
    image_paths.sort()

    def flush_batch():
        nonlocal buffer, saved
        if not buffer:
            return
        saved.extend(buffer)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(saved, f, ensure_ascii=False, indent=2)
        print(f"Batch OK — appended {len(buffer)} positives. Total saved: {len(saved)}")
        buffer = []

    processed = 0
    total_images = len(image_paths)

    for idx, img_path in enumerate(image_paths, start=1):
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Skipping unreadable image: {img_path} ({e})")
            continue

        for top_key, subcats in policy_graph.items():
            for sub_key, graph_info in subcats.items():
                prompt = build_prompt(top_key, sub_key, graph_info)
                messages = [
                    {"role": "system", "content": [{"type": "text", "text": system_rule}]},
                    {"role": "user",   "content": [
                        {"type": "image", "image": image},
                        {"type": "text",  "text": prompt},
                    ]},
                ]

                inputs = processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=120,
                        do_sample=False,
                        use_cache=True,
                    )

                resp = processor.batch_decode(
                    outputs[:, inputs["input_ids"].shape[-1]:],
                    skip_special_tokens=True
                )[0]
                clean = resp.split("<|eot|>", 1)[0].split("<|eot_id|>", 1)[0].strip()

                start = clean.find("{")
                end   = clean.rfind("}") + 1
                if start == -1 or end <= start:
                    continue
                try:
                    obj = json.loads(clean[start:end])
                except Exception:
                    continue

                if obj.get("decision") != "YES":
                    continue

                obj.pop("decision", None)
                record = {
                    "image_path": img_path,
                    "category":   obj.get("category", sub_key),
                    "rating":     obj.get("rating", graph_info.get("rating","")),
                    "rationale":  obj.get("rationale", ""),
                }

                k = key_of(record)
                if k in existing:
                    continue
                existing.add(k)
                buffer.append(record)

        processed += 1
        if processed % batch_size == 0:
            flush_batch()
            print(f"Progress: {processed}/{total_images} images processed.")

    flush_batch()
    print(f"Done. Processed {processed} images. Positives total: {len(saved)}. Output → {output_file}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--images_root", default="/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/am84fuxo/data/images")
    parser.add_argument("--output_file", default="/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/am84fuxo/data/labels/pegi/labels_all_v1.json")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--model_id", default="meta-llama/Llama-4-Scout-17B-16E-Instruct")
    parser.add_argument("--offload_dir", default="/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/am84fuxo/offload_cache")
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(args.model_id)
    model = Llama4ForConditionalGeneration.from_pretrained(
        args.model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,      
        low_cpu_mem_usage=True,
        offload_folder=args.offload_dir, 
        offload_state_dict=True,
    )
    model.eval()
    torch.set_grad_enabled(False)

    system_rule = (
        "You are a careful, literal safety reviewer. "
        "Output ONLY a single JSON object exactly matching the schema provided. "
        "No extra text, no markdown, no explanations outside JSON."
    )

    process_all_images(
        images_root=args.images_root,
        output_file=args.output_file,
        model=model,
        processor=processor,
        system_rule=system_rule,
        batch_size=args.batch_size,
    )
