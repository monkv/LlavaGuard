import json
import re
import os
import argparse
import sys
sys.path.append('/pfss/mlde/workspaces/mlde_wsp_PI_Kersting/am84fuxo/LlavaGuard')

from openai import OpenAI
from PIL import Image
from typing import Dict, Any

from llavaguard_config import local_data_dir, local_models_dir
from pegiguard.data.old.labeling import pil_to_data_url
from pegiguard.evaluation.metrics_calculator import EvaluationMetricsCalculator

def _sanitize_json_like(s: str) -> str:
    # Remove trailing commas before } or ]
    # e.g.  "Unsafe",\n}  ->  "Unsafe"\n}
    s = re.sub(r',(\s*[}\]])', r'\1', s)
    return s

def extract_json(text: str):
    """Extract JSON block from the given text and be forgiving about trailing commas."""
    # 1) Try direct JSON
    candidate = _sanitize_json_like(text)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # 2) fenced ```json ... ``` block
    fenced_pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
    m = re.search(fenced_pattern, text, flags=re.DOTALL)
    if m:
        json_str = _sanitize_json_like(m.group(1))
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            print("Failed to parse fenced JSON block:")
            print(json_str)
            raise

    # 3) first {...} substring
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and start < end:
        json_str = _sanitize_json_like(text[start:end+1])
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            print("Failed to parse extracted JSON substring:")
            print(json_str)
            raise

    print(f"****Problematic model response (no JSON found): {text}****")
    raise ValueError("No JSON block found in the text.")


def call_vllm(img_path: str, prompt: str, model, sample_id) -> Dict[str, Any]:
    '''Docstring'''
    client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="EMPTY")
    # convert image to data url (for llama4)
    image = Image.open(img_path).convert("RGB")
    data_url = pil_to_data_url(image)

    messages = [
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]},
        ]
    #print(f'****Prompt: {prompt}')
    #emperature=0.7, top_p=0.80, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
        top_p=0.80,
        extra_body={
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 1.5,
            "repetition_penalty": 1.0,
            "chat_template_kwargs": {
                "enable_thinking": False
            }
        }
    )
    raw_respnose = (resp.choices[0].message.content or "").strip()
    try:
        response = extract_json(raw_respnose)
    except Exception as e:
        print("### Error in call_vllm while parsing model output ###")
        print(f"Raw completion:\n{raw_respnose}")
        print(f"Exception: {repr(e)}")
        raise

    #print("****Model Response: ", response)
    return response


def evaluate_dataset(model_dir: str, mode: str, data_path: str, output_dir: str, responses_path: str):
    """Evaluate the model on the given dataset."""

    with open(data_path, "r", encoding = "utf-8") as f:
        data = json.load(f)

    # stuff for saving model responses
    os.makedirs(os.path.dirname(responses_path) or ".", exist_ok=True)
    batch_size = 20

    # metrics calculator initializing
    emc = EvaluationMetricsCalculator()

    # prepare data map and try to load any existing responses so we can skip them
    total = len(data)
    id_to_item = {item['id']: item for item in data}
    buffer = []
    existing_ids = set()
    if os.path.exists(responses_path):
        try:
            with open(responses_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if isinstance(existing, list):
                buffer = existing
                existing_ids = {it.get('id') for it in buffer if isinstance(it, dict) and 'id' in it}
                print(f"Loaded {len(existing_ids)} existing responses from {responses_path}")
                # feed existing responses into metrics calculator
                for it in buffer:
                    sid = it.get('id')

                    resp = it.get('model_response')
                    if sid in id_to_item and resp is not None:
                        try:
                            gt = id_to_item[sid]['conversations'][1]['value']
                            emc.add_sample(mode, sid, resp, gt)
                        except Exception as e:
                            print(f"Warning adding existing sample {sid} to metrics: {e}")
        except Exception as e:
            print(f"Warning: couldn't load existing responses from {responses_path}: {e}")
    
    print('#' * 20 + 'Starting Evaluation' + '#' * 20)
    for idx, eval_item in enumerate(data):
        # load sample
        sample_id = eval_item['id']
        gt = eval_item['conversations'][1]["value"] #ground truth
        prompt = eval_item['conversations'][0]["value"]
        image_path = eval_item['image path']

        # skip samples we've already generated responses for
        if sample_id in existing_ids:
            #print(f"Skipping sample id={sample_id} ({idx+1}/{total}) — already present in {responses_path}")
            continue
        #print(f'***Evaluating sample id: {sample_id}')

        try:
            resp = call_vllm(image_path, prompt, model_dir, sample_id)
        except Exception as e:
            import traceback
            print("\n" + "#" * 60)
            print(f"ERROR while processing sample index {idx}, id={sample_id}")
            print(f"Image path: {image_path}")
            print(f"Exception: {repr(e)}")
            traceback.print_exc()
            print("#" * 60 + "\n")
            continue

        # save model response to buffer/file?
        buffer.append({
            "id": sample_id,
            "model_response": resp
        })

        if (idx % batch_size == 0):
            with open(responses_path, "w", encoding="utf-8") as f:
                json.dump(buffer, f, ensure_ascii=False, indent=4)
            print(
                f"[Checkpoint] Wrote {idx}/{total} to {responses_path}",
                flush=True,
            )
        emc.add_sample(mode, sample_id, resp, gt)
    
    # saving model responses
    with open(responses_path, "w", encoding="utf-8") as f:
        json.dump(buffer, f, ensure_ascii=False, indent=4)
        print(f'Saved model responses to {responses_path}')
    
    metric = emc.compute(mode)

    print('#' * 20 + 'Evaluation Done' + '#' * 20)

    os.makedirs(os.path.dirname(output_dir) or ".", exist_ok=True)
    with open(output_dir, "w", encoding="utf-8") as f:
        json.dump(metric, f, ensure_ascii=False, indent=4)

    print(f'Saved evaluation results to {output_dir}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PEGI-Guard Evaluation')

    parser.add_argument('--model_dir', type=str, default= None)
    parser.add_argument('--data_path', type=str, default= None) #"Qwen/Qwen3.6-35B-A3B")
    parser.add_argument('--output_dir', type=str, default=None) # where to save the evaluation stats
    parser.add_argument('--responses_path', type=str, default=None) # where to save model responses
    parser.add_argument('--engine', type=str, default='auto')
    parser.add_argument('--batching', type=str, default='auto', choices=['auto', 'off', 'continues batching'])
    parser.add_argument('--label_mode', default='age_rating', choices=['categories', 'age_rating', 'safety']) # use different test DSs

    args = parser.parse_args()

    label_mode = args.label_mode
    
    model_dir = args.model_dir if args.model_dir is not None else "OpenGVLab/InternVL3-1B-hf" 
    
    data_path = args.data_path if args.data_path is not None else f"{local_data_dir}/datasets/PEGI-Guard-DS/age_rating/v2/test.json" 
    
    output_dir = args.output_dir if args.output_dir is not None else f"{local_data_dir}/evaluation/age_rating/base_models/OpenGVLab_InternVL3-1B-hf/test_results.json"
    
    responses_path = args.responses_path if args.responses_path is not None else f"{local_data_dir}/evaluation/age_rating/base_models/OpenGVLab_InternVL3-1B-hf/responses.json"

    print(f'*Evaluating model: {model_dir} on data: {data_path}*')
    print(f'*Output directory: {output_dir}*')
    print(f'Format is ------ {label_mode}***')

    evaluate_dataset(model_dir=model_dir, mode=label_mode, data_path=data_path, output_dir=output_dir, responses_path=responses_path)