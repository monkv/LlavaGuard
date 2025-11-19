import os
import json
import argparse
from openai import OpenAI
from PIL import Image
import random
import textwrap
import sys
sys.path.append('/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/am84fuxo/LlavaGuard')
from llavaguard.taxonomy.PEGI.PEGI_Graph import policy_graph, policy_graph_to_safety_policy, get_max_pegi_score, policy_graph_to_rationale
from llavaguard_config import local_data_dir
from pegiguard.data.labeling import pil_to_data_url
import textwrap, json, re

MODEL_ID = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
CHECK_POINT = 50
#SYSTEM_RULE_1 = ("""
#                You are shown an image along with its assigned safety categories (there may be none).
#                Your task is to explain concisely and factually why the image relates to the given categories and explain the review.
#                If one or more assigned categories do not clearly apply, omit them rather than contradicting or speculating.
#               
#            """
#               )
#SYSTEM_RULE_2 = ("""
#                You are shown an image along with its assigned safety categories (there may be none).
#                Your task is to explain concisely and factually why the image relates to the given categories.
#            """
#               )
#SYSTEM_RULE_3 = (""" You are a careful reviewer.
#            """
#               )


def build_prompt_strict(pegi_score: str, cats: list[str], policy_text: str):
    cats_json = json.dumps(cats, ensure_ascii=False)
    prompt = f"""
You are a literal, rule-bound safety analyst.

PEGI {pegi_score} policy (for reference only):
{policy_text}

TASK:
- Analyze the image and ONLY write a single short, factual, supportive sentence for EACH category in the provided list.
- Treat EVERY provided category as correct; DO NOT negate, question, or exclude any category.
- DO NOT add categories that are not provided.
- DO NOT make a policy judgment; just describe how the image relates to each category.
- Write plain text only.

Rules:
- "explain" must be ≤ 25 words, **no negations** (no "does not relate/depict", "there is no evidence" etc.).
- Include ALL provided categories, in the SAME order as given.

Categories (use these verbatim and ONLY these):
{cats_json}
"""
    return textwrap.dedent(prompt).strip()


def get_user_age(idx: int):
    """
    Returns a random user age that is allowed for a certain PEGI score.

    PEGI levels by index:
        0 -> PEGI 0 allowed ages 0-2
        1 -> PEGI 3 allowed ages 3-6
        2 -> PEGI 7 allowed ages 7-11
        3 -> PEGI 12 allowed ages 12-15
        4 -> PEGI 16 allowed ages 16-17
        5 -> PEGI 18 allowed ages 18+
    """
    age_ranges = [
        (0, 2),      # idx 0: 0–2
        (3, 6),      # idx 1: 3–6
        (7, 11),     # idx 2: 7–11
        (12, 15),    # idx 3: 12–15
        (16, 17),    # idx 4: 16–17
        (18, None)   # idx 5: 18+
    ]

    if idx < 0 or idx >= len(age_ranges):
        raise ValueError(f"idx must be between 0 and {len(age_ranges) - 1}")

    low, high = age_ranges[idx]

    if high is None:
        # PEGI 18 = any adult age (e.g. 18–99)
        return random.randint(low, 99)
    else:
        return random.randint(low, high)

def get_pegi_rating(index: int) -> int:
    """Return PEGI rating number given an index (0–5)."""
    mapping = {
        0: 0,
        1: 3,
        2: 7,
        3: 12,
        4: 16,
        5: 18
    }
    return mapping.get(index, None)

def get_rationale(client, img_path: str, assessment: str, categories: str,  subcategories: list, pegi_score: str):
    """future docstring"""
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
    image = Image.open(img_path).convert("RGB")
    data_url = pil_to_data_url(image)

    if subcategories:
        cats_str = "\n".join(f"- {c}" for c in subcategories)
    else:
        cats_str = "NA: None applicable"
    rationale_policy = policy_graph_to_rationale(pegi_score) #2
    #print(f"POLICY: {rationale_policy})
    #prompt_1 = f"Describe the image and how it relates to the following safety-categories in maximum of 4 sentences (less is allowed) :\n{cats_str}"

    #prompt_2 = f"""You are a content safety analyst. Analyze the given image carefully.
    #            1. First, briefly describe what is shown in the image (1-2 sentences).
    #            2. Then, explain how the image relates to each of the following safety categories (up to 4 sentences total):
    #
    #           {cats_str}
    #
    #           These categories come from the following graph-based safety policy:
    #           {rationale_policy}
    #
    #           The policy has been adjusted to match the user's age - {user_age} and their corresponding PEGI rating. 
    #
    #           Write clear, factual rationales that explain why the image fits the given categories. Do not mention any other categories.
    #          
    #         If there no categories the image relates to, explain why the image is safe.
    #
    #            If one of the categoies is in "prohibited content categories" for this age group, explain that the content is unsafe.
    #
    #           Do not use any fancy markdown, just write a simple text - image discription and explanation of relationship to the given categories.
    #           """
    if assessment == "Unsafe":
        #prompt = build_prompt_strict(pegi_score, subcategories, rationale_policy)
        prompt = f"""Consider the following policy for PEGI {pegi_score}:
        {rationale_policy}
        
        You are a content safety analyst preparing rationales for a labelled dataset. 
        The human label and assessment provided below are authoritative and must not be questioned or contradicted.
        Your task: Write a short rationale that *supports and justifies* the given assessment using only the visible evidence in the image.

        Follow these steps exactly:
        1. Image description (1-2 sentences): Describe what is shown in the image using clear, neutral language.
        2. Category Analysis (up to 4 sentences total): For each of the following safety categories, briefly explain how the image relates to it. If a category does not apply, SKIP it! DO NOT explain that it it doesn't relate/doesn't depict the category! 
        Do NOT mention uncertainty, missing evidence, or potential alternative interpretations.
        {cats_str}. 
        3. Explain why this content was marked as {assessment} for the PEGI {pegi_score}: State which of the above categories violate the PEGI {pegi_score} policy (are in the "prohibited content categories" list).

        Hard constraints:
        - Never contradict or question the given assessment or categories, never imply that is is wrong
        - Never introduce or assess new categories besides the given ones.
        - Write the categories in the rationale exactly as they appear in the policy, i.e. with numbers.
        - Do NOT include section labels (e.g., "Image description:", "Category analysis:"), bullet points, lists
        """
    elif not (subcategories == "NA: None applying"):

        prompt = f""""Consider the following policy for PEGI {pegi_score}:
        {rationale_policy}
        
        You are a content safety analyst preparing rationales for a labelled dataset. 
        The human label and assessment provided below are authoritative and must not be questioned or contradicted.
        Your task: Write a short rationale that *supports and justifies* the given assessment using only visible evidence in the image.

        Follow these steps exactly:
        1. Image description (1-2 sentences): Describe what is shown in the image in clear, neutral language.
        2. Up to 4 sentences total: The image was assessed with the following safety categories. For each of the categories, briefly explain how the image relates to them. If a category doesn't apply, SKIP it! DO NOT explain that the image doesn't relate to/doesn't depict the category! 
        Do NOT mention uncertainty, missing evidence, or potential alternative interpretations. 
        {cats_str}. 
        3. There are no related category that violate the PEGI {pegi_score} policy (i.e. that are on the "prohibited content categories" list). Explain why this image is complient with the PEGI {pegi_score} policy and is considered safe users over {pegi_score} age.
        
        Hard constraints:
        - Never contradict or question the given assessment or categories.
        - Never introduce or assess new categories besides the given ones.
        - Write the categories in the rationale exactly as they appear in the policy, i.e. with numbers.
        - Stay concise and factual.
        - Do NOT include section labels (e.g., "Image description:", "Category analysis:"), bullet points, lists.
        """
        #no ctageory was violated, explain that its complient
    else:
        prompt = f""""Consider the following policy for PEGI {pegi_score}:
        {rationale_policy}

        You are a content safety analyst preparing rationales for a labelled dataset. 
        The human label and assessment provided below are authoritative and must not be questioned or contradicted.
        Your task: Write a short rationale that *supports and justifies* the given assessment using only visible evidence in the image.

         Follow these steps exactly:
        1. Image description (1-2 sentences): Describe what is shown in the image in clear, neutral language.
        2. There are no related safety categories for this image. Use this fact to explain why this image is complient with the PEGI {pegi_score} policy and is considered safe users over {pegi_score} age.
        
        Hard constraints:
        - Never contradict or question the given assessment or categories.
        - Never introduce or assess new categories besides the given ones.
        - Write the categories in the rationale exactly as they appear in the policy, i.e. with numbers.
        - Stay concise and factual.
        - Do NOT include section labels (e.g., "Image description:", "Category analysis:"), bullet points, lists.
        """
    #print(f"\n PROMPT: \n {prompt}")
    messages = [
        #{"role": "system", "content": SYSTEM_RULE_1},
        {"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]},
    ]

    resp = client.chat.completions.create(
        model=MODEL_ID,
        messages=messages,
    )
    response = resp.choices[0].message.content.strip()
    #print("--------------------------------------")
    #print(f"Age:  {user_age}")
    #print(f"Categories:  {subcategories}")
    #print(f"Assessment: {assessment}")
    #print(f"PEGI {pegi_score} policy")
    #print(f"RESPONSE:{response}")

    gpt_value = json.dumps({
        "categories":  categories,
        "subcategories":  subcategories,
        "rationale": response,
        "assessment": assessment,
    }, indent=4)


    return gpt_value


def create_pegi_dataset(input_file: str, output_file: str, batch_size: int = 10):
    """
    docstring incoming
    """

    with open(input_file, "r", encoding = "utf-8") as f:
        origin_data = json.load(f)
   
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    total = len(origin_data)
    total_out = 0
    buffer = []

    client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
    # iterate over every image in the dataset

    for sample_idx, sample in enumerate(origin_data, start=1):
        # extract all the info from the standart ds
        img_id = sample["id"]
        img_path = sample["image_path"]
        categories = sample["categories"]
        subcategories = sample["subcategories"]

        pegi_score = get_max_pegi_score(sample["subcategories"])
        range_high = 1 if pegi_score == 0 else 6 #decides if 1 sample or 6 have to be made

        # adding 6 samples of the same image for 6 different age groups (bzw just 1 for completely safe images)
        for pegi_idx in range(0,range_high):
            user_pegi = get_pegi_rating(pegi_idx)
            user_age = get_user_age(pegi_idx) #returns a random age for the spesific group
            assessment = "Safe" if user_pegi >= pegi_score else "Unsafe"
            augmented_policy = policy_graph_to_safety_policy(user_pegi)
            #print(f"FROM HUMAN: \n {augmented_policy}")
            # generate a rationale
            if subcategories == []:
                categories = "NA: None applying"
                subcategories = "NA: None applying"
            gpt_value = get_rationale(client, img_path, assessment, categories, subcategories, user_pegi)
            # create an entry for the new ds

            record = {
                "id": f"{img_id}_{pegi_idx}",
                "image path": img_path,
                "categories": categories,
                "subcategories": subcategories,
                "user age": user_age,
                "assessment": assessment,
                "conversations": [
                    {"from": "human", "value": augmented_policy},
                    {"from": "gpt",   "value": gpt_value}
                ]
            }
            #print("----------------------------------")
            buffer.append(record)
            total_out = total_out + 1

        if (total_out % batch_size == 0):
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(buffer, f, ensure_ascii=False, indent=4)
            print(
                f"[Checkpoint] Wrote {total_out}/{total}*1-6 entries to {output_file}",
                flush=True,
            )
    # if some more weren't flushed
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(buffer, f, ensure_ascii=False, indent=4)

    print(f"\nDone. Wrote {len(buffer)} entries to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default=f'{local_data_dir}/PEGIGuard-DS/v3/pegiguard_no_dupl.json', help="Path to input directory")
    parser.add_argument('--output_path', type=str, default=f'{local_data_dir}/PEGIGuard-DS/v3/pegi_ds_x6_all_data.json', help="Path to output directory")
    parser.add_argument('--batch_size', type=str, default=CHECK_POINT, help="How many entries to write into a file at a time")
    args = parser.parse_args()

    create_pegi_dataset(
        input_file = args.input_path,
        output_file = args.output_path,
        batch_size= args.batch_size
    )