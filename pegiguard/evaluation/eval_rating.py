"""Generate model responses or calculate metrics from saved responses."""

import argparse
import base64
import io
import json
import os
from pathlib import Path
import re
import sys
import tempfile

# Support running this file directly from outside the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pegiguard.evaluation.metrics_calculator import EvaluationMetricsCalculator, validate_ground_truth


def _sanitize_json_like(text):
    """Remove trailing commas outside quoted JSON strings."""
    return re.sub(r'("(?:\\.|[^"\\])*"|,\s*(?=[}\]]))',
                  lambda match: match.group(0) if match.group(0).startswith('"') else '', text)


def extract_json(text):
    """Extract a response object from JSON, a fenced block, or surrounding prose."""
    if not isinstance(text, str):
        raise ValueError("Expected model output to be text")
    candidate = _sanitize_json_like(text.strip())
    try:
        result = json.loads(candidate)
    except json.JSONDecodeError:
        result = None
    else:
        if not isinstance(result, dict):
            raise ValueError("Model response must be a JSON object")
        return result
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", candidate):
        try:
            result, _ = decoder.raw_decode(candidate, match.start())
        except json.JSONDecodeError:
            continue
        if isinstance(result, dict):
            return result
    raise ValueError("No valid JSON object found in model output")


def call_vllm(img_path, prompt, model, sample_id):
    """Request one prediction from the local vLLM server."""
    # Saved-response evaluation does not need inference dependencies or a server.
    from openai import OpenAI
    from PIL import Image

    with Image.open(img_path) as image:
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG")
    data_url = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
    client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="EMPTY")
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]}],
        temperature=0.7,
        top_p=0.80,
        extra_body={
            "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5,
            "repetition_penalty": 1.0, "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    return extract_json(response.choices[0].message.content or "")


def _write_json(path, payload):
    """Replace a JSON file atomically after a successful write."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
            temporary = handle.name
            json.dump(payload, handle, ensure_ascii=False, indent=4)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def evaluate_dataset(model_dir, mode, data_path, output_dir, responses_path, responses_only=False):
    """Score one response per dataset ID and optionally generate missing responses.

    Invalid rating/assessment objects are retained and counted as invalid model
    predictions. Null or non-object responses are incomplete and may be retried.
    In responses-only mode the cache is never changed and inference is never called.
    """
    paths = [Path(path).resolve() for path in (data_path, output_dir, responses_path)]
    if len(set(paths)) != len(paths):
        raise ValueError("Dataset, metrics, and response paths must be different")
    with open(data_path, encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("Dataset must be a list")
    items = {}
    ground_truths = {}
    if mode not in {"age_rating", "safety"}:
        raise ValueError(f"Unsupported evaluation mode: {mode!r}")
    # Validate references before inference or any output writes.
    for item in data:
        if not isinstance(item, dict) or item.get("id") is None:
            raise ValueError("Each dataset item must have an ID")
        key = str(item["id"])
        if key in items:
            raise ValueError(f"Duplicate dataset ID: {key}")
        ground_truths[key] = validate_ground_truth(mode, item["conversations"][1]["value"])
        items[key] = item

    records = {}
    if os.path.exists(responses_path):
        # A corrupt cache must not be silently overwritten by a new run.
        with open(responses_path, encoding="utf-8") as handle:
            existing = json.load(handle)
        if not isinstance(existing, list):
            raise ValueError("Saved responses must be a list")
        for record in existing:
            if not isinstance(record, dict) or record.get("id") is None:
                raise ValueError("Each saved response must have an ID")
            key = str(record["id"])
            previous = records.get(key)
            # Prefer the last object response; an incomplete duplicate cannot erase it.
            if previous is None or isinstance(record.get("model_response"), dict) or not isinstance(previous.get("model_response"), dict):
                records[key] = record

    calculator = EvaluationMetricsCalculator()
    missing_ids = []
    failed_ids = []
    generated = 0
    for key, item in items.items():
        response = records.get(key, {}).get("model_response")
        if not isinstance(response, dict):
            if responses_only:
                missing_ids.append(item["id"])
                continue
            try:
                image_path = item.get("image path") or item.get("image_path") or item.get("image")
                if not image_path:
                    raise ValueError("Missing image path")
                response = call_vllm(image_path, item["conversations"][0]["value"], model_dir, item["id"])
                if not isinstance(response, dict):
                    raise ValueError("Model response must be a JSON object")
            except Exception as error:
                print(f"Failed sample {item['id']}: {error}")
                failed_ids.append(item["id"])
                continue
            records[key] = {"id": item["id"], "model_response": response}
            generated += 1
            if generated % 20 == 0:
                _write_json(responses_path, list(records.values()))
        calculator.add_sample(mode, item["id"], response, ground_truths[key])

    if not responses_only:
        _write_json(responses_path, list(records.values()))
    metrics = calculator.compute(mode)
    metrics.update({
        "dataset_samples": len(items),
        "missing_response_ids": missing_ids,
        "failed_response_ids": failed_ids,
        "complete": not missing_ids and not failed_ids,
    })
    _write_json(output_dir, metrics)
    print(f"Saved metrics to {output_dir}; generated {generated} responses; "
          f"missing {len(missing_ids)}, failed {len(failed_ids)}")
    return metrics


def main():
    """Parse supported evaluation options and resolve default data paths."""
    parser = argparse.ArgumentParser(description="PEGI-Guard evaluation")
    parser.add_argument("--model_dir", default="OpenGVLab/InternVL3-1B-hf")
    parser.add_argument("--data_path")
    parser.add_argument("--output_dir", help="Output metrics JSON file")
    parser.add_argument("--responses_path")
    parser.add_argument("--label_mode", default="age_rating", choices=["age_rating", "safety"])
    parser.add_argument("--responses_only", action="store_true", help="Score saved responses without calling a model")
    args = parser.parse_args()
    if not all((args.data_path, args.output_dir, args.responses_path)):
        from llavaguard_config import local_data_dir
        model_name = re.sub(r"[/:]", "_", args.model_dir)
        result_dir = f"{local_data_dir}/evaluation/{args.label_mode}/base_models/{model_name}"
        args.data_path = args.data_path or f"{local_data_dir}/datasets/PEGI-Guard-DS/{args.label_mode}/v2/test.json"
        args.output_dir = args.output_dir or f"{result_dir}/test_results.json"
        args.responses_path = args.responses_path or f"{result_dir}/responses.json"
    evaluate_dataset(args.model_dir, args.label_mode, args.data_path, args.output_dir,
                     args.responses_path, responses_only=args.responses_only)


if __name__ == "__main__":
    main()
