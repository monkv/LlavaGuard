"""Build age-rating instruction data with model-generated rationales."""

import argparse
import json
import logging
from pathlib import Path
import sys

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from llavaguard.taxonomy.PEGI.PEGI_Graph import get_max_pegi_score, policy_graph_to_rating_policy
from pegiguard.data.dataset_utils import (
    image_path_of, labels_to_string, pil_to_data_url, prepare_build, sample_labels, write_records,
)

MODEL_ID = "Qwen/Qwen3.6-35B-A3B"
CHECK_POINT = 10
logger = logging.getLogger(__name__)

def get_pegi_output(client, img_path: str, categories: str, subcategories: list) -> str:
    """
    Generate the expected output for the model: categories, subcategories, PEGI rating, and rationale.

    Args:
        client client instance
        img_path: Path to the image file
        categories: Category string
        subcategories: List of subcategories

    Returns:
        JSON string with categories, subcategories, PEGI rating, and rationale
    """
    try:
        with Image.open(img_path) as image:
            data_url = pil_to_data_url(image)
    except Exception as e:
        logger.error(f"Error loading image {img_path}: {e}")
        raise

    pegi = get_max_pegi_score(subcategories)

    if subcategories:
        prompt = f"""Given that this image relates to the following safety subcategories: {', '.join(subcategories)}, with overall PEGI rating {pegi}, provide a rationale explaining why the image relates to those safety subcategories and how the rating is assessed.

Instructions:
- Start with 1–2 sentences describing the image.
- Then, explain how the image relates to each of the listed subcategories.
- Conclude with the final PEGI rating.

Constraints:
- Do not contradict the given subcategories or rating.
- Do not introduce new categories.
- Use subcategory names exactly as written. Do not use extra formatting like ** or \\
- Keep it concise.
"""
    else:
        prompt = f"""This image relates to no safety categories, so the overall PEGI rating is 3 (safe for all).

Provide a rationale explaining why the image is safe and relates to no safety categories.

Instructions:
- Start with 1–2 sentences describing the image.
- Explain why the image is overall safe (do not cite all of the categories from the policy).
- Confirm that the PEGI rating is 3.

Constraints:
- Do not introduce categories.
- Keep it concise.
"""

    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]},
    ]

    try:
        resp = client.chat.completions.create(
            model=MODEL_ID,
            messages=messages,
        )
        response = (resp.choices[0].message.content or "").strip()
        if not response:
            raise ValueError("Empty rationale response")
    except Exception as e:
        logger.error(f"Error calling OpenAI API for image {img_path}: {e}")
        raise

    output = {
        "categories": categories,
        "subcategories": labels_to_string(subcategories),
        "PEGI rating": str(pegi),
        "rationale": response,
    }
    return json.dumps(output, indent=4, ensure_ascii=False)


def create_pegi_v2_dataset(input_file, output_file, batch_size=10, ids_to_process=None):
    """Build all records or selected IDs, skipping already saved record IDs."""
    origin_data, records = prepare_build(input_file, output_file, batch_size)
    existing_ids = {str(record['id']) for record in records}
    requested = None if ids_to_process is None else {str(value) for value in ids_to_process}
    candidates = [sample for sample in origin_data if requested is None or str(sample['id']) in requested]
    if requested is not None:
        missing = requested - {str(sample['id']) for sample in candidates}
        if missing:
            raise ValueError(f"Requested IDs not found: {sorted(missing)}")
    pending = [sample for sample in candidates if str(sample['id']) not in existing_ids]
    client = None
    generated = 0
    policy = policy_graph_to_rating_policy()
    for sample in pending:
        categories, subcategories = sample_labels(sample)
        image_path = image_path_of(sample)
        if client is None:
            from openai import OpenAI
            client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
        try:
            answer = get_pegi_output(client, image_path, categories, subcategories)
        except Exception as error:
            logger.error("Could not generate rationale for %s: %s", sample['id'], error)
            continue
        records.append({
            'id': sample['id'], 'image path': image_path,
            'categories': categories, 'subcategories': labels_to_string(subcategories),
            'age_rating': get_max_pegi_score(subcategories),
            'conversations': [{'from': 'human', 'value': policy}, {'from': 'gpt', 'value': answer}],
        })
        generated += 1
        if generated % batch_size == 0:
            write_records(output_file, records)
    write_records(output_file, records)
    logger.info("Saved %s records to %s", len(records), output_file)


def create_pegi_v2_dataset_for_ids(input_file, output_file, ids_to_process, batch_size=10):
    """Build only the requested IDs using the same resume behavior."""
    return create_pegi_v2_dataset(input_file, output_file, batch_size, ids_to_process)


if __name__ == '__main__':
    from llavaguard_config import local_data_dir
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    parser = argparse.ArgumentParser(description='Build PEGI age-rating data')
    parser.add_argument('--input_path', default=f'{local_data_dir}/datasets/PEGI-Guard-DS/v4/all_labels.json')
    parser.add_argument('--output_path', default=f'{local_data_dir}/datasets/PEGI-Guard-DS/v4/age_rating/all_data.json')
    parser.add_argument('--batch_size', type=int, default=CHECK_POINT)
    parser.add_argument('--ids', nargs='+', help='Optional subset of record IDs')
    args = parser.parse_args()
    create_pegi_v2_dataset(args.input_path, args.output_path, args.batch_size, args.ids)
