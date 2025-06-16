import os
import argparse
import glob
import sys
sys.path.append('/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/am84fuxo/LlavaGuard')
import torch
from transformers import AutoProcessor, Llama4ForConditionalGeneration
from PIL import Image
from llavaguard_config import local_image_dirs, local_data_dir

import pandas as pd
import json


MODEL_ID = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
CHECK_POINT = 100  # write partial file every 100 images

def main(args):
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Llama4ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    csv_path = f"{local_data_dir}/data/annotations/auto_generated_annotations/v0_2/extracted_categories_stripped.csv"
    output_path = args.output_path
    df = pd.read_csv(csv_path)
    total = len(df)
    print(f"Loaded {len(df)} rows from:\n  {csv_path}")
    #df = df.tail(5)
    all_outputs = []

    for idx, row in df.iterrows():
        sample_id = row["image_name"]
        #print(row["file_path"])
        img_path  = row["file_path"]
        category  = row["category"]

        prompt = (
            f"Describe the image and how it relates to the following PEGI rating safety-category - {category} - in 2 sentences."
        )
        print(f"\n[Sample {idx + 1} / {len(df)}")
        #print(img_path)
        image = Image.open(img_path).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        outputs = model.generate(**inputs, max_new_tokens=200)

        start = inputs["input_ids"].shape[-1]
        rationale_text = processor.batch_decode(outputs[:, start:])[0].strip()
        #print("  →  Generated rationale:")
        #print(f"    {rationale_text}")

        all_outputs.append({
            "image_name": sample_id,
            "image_path": img_path,
            "category":   category,
            "rationale":  rationale_text
        })
        if (idx + 1) % CHECK_POINT == 0 or idx + 1 == total:
            parent_dir = os.path.dirname(output_path)
            os.makedirs(parent_dir, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(all_outputs, f, indent=4)
            print(f"  → [Checkpoint] Wrote {idx+1} / {total} entries", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, 
                        default=f'{local_data_dir}/data/rationales', help="Path to output directory")
    args = parser.parse_args()
    main(args)
