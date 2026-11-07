"""Build age-dependent safety instruction data with model-generated rationales."""

import argparse
import json
import logging
import random
from pathlib import Path
import sys

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from llavaguard.taxonomy.PEGI.PEGI_Graph import get_max_pegi_score, policy_graph_to_safety_policy, policy_graph_to_rationale
from pegiguard.data.dataset_utils import (
    image_path_of, labels_to_string, pil_to_data_url, prepare_build, sample_labels, write_records,
)

MODEL_ID = "Qwen/Qwen3.6-35B-A3B"
CHECK_POINT = 5
logger = logging.getLogger(__name__)

def get_user_age(idx: int) -> int:
    """
    Returns a random user age that is allowed for a certain PEGI score.

    PEGI levels by index:
        0 -> PEGI 3 allowed ages 3-6
        1 -> PEGI 7 allowed ages 7-11
        2 -> PEGI 12 allowed ages 12-15
        3 -> PEGI 16 allowed ages 16-17
        4 -> PEGI 18 allowed ages 18+
    """
    age_ranges = [
        (3, 6),      # Idx 0: 3–6.
        (7, 11),     # Idx 1: 7–11.
        (12, 15),    # Idx 2: 12–15.
        (16, 17),    # Idx 3: 16–17.
        (18, None)   # Idx 4: 18+.
    ]

    if idx < 0 or idx >= len(age_ranges):
        raise ValueError(f"idx must be between 0 and {len(age_ranges) - 1}")

    low, high = age_ranges[idx]

    if high is None:
        # PEGI 18 = any adult age (e.g. 18–99).
        return random.randint(low, 99)
    else:
        return random.randint(low, high)


def get_pegi_rating(index: int) -> int:
    """Return PEGI rating number given an index (0–4)."""
    mapping = {
        0: 3,
        1: 7,
        2: 12,
        3: 16,
        4: 18
    }
    if index not in mapping:
        raise ValueError(f"Invalid index {index}. Must be between 0 and 4.")
    return mapping[index]

def get_rationale(client, img_path: str, assessment: str, categories: str, subcategories: list, pegi_score: int) -> str:
    """
    Generate a rationale for the image assessment using the OpenAI API.

    Args:
        client client instance
        img_path: Path to the image file
        assessment: "Safe" or "Unsafe"
        categories: Category string
        subcategories: List of subcategories
        pegi_score: PEGI score number

    Returns:
        JSON string with categories, subcategories, rationale, and assessment
    """
    try:
        with Image.open(img_path) as image:
            data_url = pil_to_data_url(image)
    except Exception as e:
        logger.error(f"Error loading image {img_path}: {e}")
        raise

    if subcategories:
        cats_str = "\n".join(f"- {c}" for c in subcategories)
    else:
        cats_str = "NA: None applying"

    rationale_policy = policy_graph_to_rationale(pegi_score)

    # Build prompt based on assessment and categories.
    base_prompt = f"""Consider the following policy for PEGI {pegi_score}:
{rationale_policy}

You are a content safety analyst writing rationales for a labelled dataset.
The provided labels and assessment are authoritative and must not be questioned.
The PEGI policy text above is the only source of truth for whether a category violates this rating.

Write a short rationale using only visible evidence in the image.

Instructions:
- Start with 1–2 sentences describing the image.
- Then, in up to 4 sentences total, explain how the image relates to these categories (only include the given ones, do not mention other ones):
{cats_str}

Constraints:
- Do not contradict the given categories or assessment.
- Do not introduce new categories.
- Use category names exactly as written.
- Do not mention uncertainty or alternative interpretations.
- No section labels or lists.
- Keep it concise.
"""
    # Build prompt based on assessment and categories.
    if assessment == "Unsafe":
        prompt = base_prompt + f"""
Conclude by naming each of the listed categories that violate the PEGI {pegi_score} policy (categories that are marked as prohibited in the policy).
If a listed category is prohibited under PEGI {pegi_score}, do not say it is allowed.
If multiple listed categories violate the policy, include them all in your conclusion.
"""
    elif subcategories:
        prompt = base_prompt + f"""
Conclude by explaining why none of these categories violate the PEGI {pegi_score} policy and why the image is safe for users aged {pegi_score}+."""
    else:
        prompt = f"""Consider the following policy for PEGI {pegi_score}:
{rationale_policy}

You are a content safety analyst writing rationales for a labelled dataset.
The provided label and assessment are authoritative.

Describe the image in 1–2 sentences, then explain briefly why the absence of relevant safety categories makes it compliant with PEGI {pegi_score}.

Constraints:
- Do not contradict the given categories or assessment.
- Do not introduce categories.
- No section labels or lists.
- Keep it concise.
"""

    messages: list[dict] = [
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

    result = {
        "categories": categories,
        "subcategories": labels_to_string(subcategories),
        "rationale": response,
        "assessment": assessment,
    }
    return json.dumps(result, indent=4, ensure_ascii=False)


def create_pegi_dataset(input_file, output_file, batch_size=10):
    """Resume each image/age variant independently, including partial images."""
    origin_data, records = prepare_build(input_file, output_file, batch_size)
    existing_ids = {str(record['id']) for record in records}
    client = None
    generated = 0
    for sample in origin_data:
        categories, subcategories = sample_labels(sample)
        image_path = image_path_of(sample)
        rating = get_max_pegi_score(subcategories)
        # Safe-for-all images need only the lowest-age variant.
        for pegi_index in range(1 if rating == 3 else 5):
            record_id = f"{sample['id']}_{pegi_index}"
            if record_id in existing_ids:
                continue
            age = get_pegi_rating(pegi_index)
            assessment = 'Safe' if age >= rating else 'Unsafe'
            if client is None:
                from openai import OpenAI
                client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
            try:
                answer = get_rationale(client, image_path, assessment, categories, subcategories, age)
            except Exception as error:
                logger.error("Could not generate rationale for %s: %s", record_id, error)
                continue
            records.append({
                'id': record_id, 'image path': image_path, 'categories': categories,
                'subcategories': labels_to_string(subcategories), 'age_rating': rating,
                'user age': age, 'assessment': assessment,
                'conversations': [
                    {'from': 'human', 'value': policy_graph_to_safety_policy(age)},
                    {'from': 'gpt', 'value': answer},
                ],
            })
            existing_ids.add(record_id)
            generated += 1
            if generated % batch_size == 0:
                write_records(output_file, records)
    write_records(output_file, records)
    logger.info("Saved %s records to %s", len(records), output_file)


if __name__ == '__main__':
    from llavaguard_config import local_data_dir
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    parser = argparse.ArgumentParser(description='Build PEGI safety data')
    parser.add_argument('--input_path', default=f'{local_data_dir}/datasets/PEGI-Guard-DS/v4/all_labels.json')
    parser.add_argument('--output_path', default=f'{local_data_dir}/datasets/PEGI-Guard-DS/v4/safety/all_data.json')
    parser.add_argument('--batch_size', type=int, default=CHECK_POINT)
    args = parser.parse_args()
    create_pegi_dataset(args.input_path, args.output_path, args.batch_size)
