# *************************************************************************
# This file may have been modified by Bytedance Inc. (“Bytedance Inc.'s Mo-
# difications”). All Bytedance Inc.'s Modifications are Copyright (2025) B-
# ytedance Inc..  
# *************************************************************************

# Adapted from https://github.com/NVlabs/describe-anything/blob/main/evaluation/eval_model_outputs.py

# Copyright 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import base64
import io
import inflect
import json
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pycocotools import mask as mask_utils

import openai
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm

# Define Azure OpenAI details
model_name = "gpt-5.2-2025-12-11"
max_tokens = 1000  # range: [1, 4095]

# Initialize the Azure client
client = openai.AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version=model_name,
)

prompt_eval = """Answer the multiple-choice question based on the text description of an object in this image. You need to follow these rules:
1. Do not output any reasoning. Do not perform correction. Please output exactly one answer from the choices for each question. Do not repeat the question.
2. There is no need for exact matching. Please choose the closest option based on the description.

The description is:
{pred_caption}

From the description above, please answer the following question with one of the choices:
{question_text_str}
"""

api_call_count = 0
api_call_lock = threading.Lock()
DEFAULT_CONCURRENCY = 10

def decode_rle(rle_dict: dict, h: int, w: int) -> np.ndarray:
    rle = {"size": [h, w], "counts": rle_dict["counts"]}
    if isinstance(rle["counts"], list):
        rle = mask_utils.frPyObjects(rle, h, w)
    return mask_utils.decode(rle).astype(np.uint8)

def query(prompt, images, temperature, max_tokens, max_retries=5):
    global api_call_count
    with api_call_lock:
        if api_call_count >= args.api_call_limit:
            raise Exception("API call limit reached")
        api_call_count += 1

    content = [
        {"type": "text", "text": "The image:\n"},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{images[0]}"}},
        {"type": "text", "text": "\nThe mask of the image:\n"},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{images[1]}"}},
        {"type": "text", "text": f"\n{prompt}\n"},
    ]

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": content}],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Error querying OpenAI API (attempt {attempt + 1}/{max_retries}): {e}. Retrying...")
                time.sleep(2 ** attempt)
            else:
                print(f"Failed to query OpenAI API after {max_retries} attempts.")
                raise e


def parse_pred(pred, choices, key):
    pred = pred.strip().lower()
    substr_indices = []
    for index, choice in enumerate(choices):
        choice = choice.strip().lower()
        prefix = "abcde"[index]
        if choice == pred or pred == f"{prefix}. {choice}" or pred == prefix:
            return index
        if choice in pred:
            substr_indices.append((index, pred.index(choice), len(choice)))

    if len(substr_indices) == 1:
        return substr_indices[0][0]

    choices_label = "abcde"
    if len(pred) >= 2 and pred[0] in choices_label and pred[1] == ".":
        return choices_label.index(pred[0])

    if substr_indices:
        if len(substr_indices) > 1:
            ret, ret_pos, _ = max(substr_indices, key=lambda x: x[1])
            max_items = [item for item in substr_indices if item[1] == ret_pos]
            if len(max_items) > 1:
                ret = max(max_items, key=lambda x: x[2])[0]
            return ret
        else:
            return substr_indices[0][0]

    match_lengths = []
    for index, choice in enumerate(choices):
        choice = choice.strip().lower()
        if pred in choice:
            match_lengths.append((index, len(choice)))
    if match_lengths:
        if len(match_lengths) > 1:
            ret = max(match_lengths, key=lambda x: x[1])[0]
        else:
            ret = match_lengths[0][0]
        return ret

    if pred and pred[0] in "abcde" and (len(pred.strip()) == 1 or (len(pred) > 1 and pred[1] == "\n")):
        return "abcde".index(pred[0])

    return None


def evaluate(question_dicts, pred_caption, temperature, max_tokens, images,
             *, response_override=None, key=None, verbose=False, executor=None):
    pred_answers = []
    prompt = []
    response = []

    prompts = []
    for question_dict in question_dicts:
        question_text_str = f"{question_dict['question']}\n"
        for choice_index, (choice, score) in enumerate(question_dict['choices']):
            question_text_str += f"{'ABCDE'[choice_index]}. {choice}\n"
        prompts.append(prompt_eval.format(
            pred_caption=pred_caption,
            question_text_str=question_text_str.strip(),
        ))

    futures = {}
    use_executor = executor is not None

    if use_executor:
        for idx, prompt_item in enumerate(prompts):
            if response_override is not None and idx < len(response_override) and response_override[idx] is not None:
                response.append(response_override[idx])
                pred_answers.append(response_override[idx].strip())
                prompt.append(prompt_item)
            else:
                fut = executor.submit(query, prompt_item, images, temperature, max_tokens)
                futures[fut] = (idx, prompt_item)
    else:
        for idx, prompt_item in enumerate(prompts):
            if response_override is not None and idx < len(response_override) and response_override[idx] is not None:
                response_item = response_override[idx]
            else:
                response_item = query(prompt_item, images, temperature, max_tokens)
            pred_answers.append(response_item.strip())
            prompt.append(prompt_item)
            response.append(response_item)

    if futures:
        total = len(prompts)
        pred_answers_by_index = [None] * total
        response_by_index = [None] * total
        prompt_by_index = prompts[:]

        if response_override is not None:
            for idx in range(len(prompts)):
                if idx < len(response_override) and response_override[idx] is not None:
                    response_by_index[idx] = response_override[idx]
                    pred_answers_by_index[idx] = response_override[idx].strip()

        for fut in as_completed(list(futures.keys())):
            idx, prompt_item = futures[fut]
            try:
                resp = fut.result()
            except Exception as e:
                print(f"Error in concurrent query for key {key}, question idx {idx}: {e}")
                raise
            response_by_index[idx] = resp
            pred_answers_by_index[idx] = resp.strip()

        prompt = prompt_by_index
        response = response_by_index
        pred_answers = pred_answers_by_index

    # Parse answers and compute scores
    pred_indices = [
        parse_pred(pred_answer, [c for c, s in qd['choices']], key)
        for pred_answer, qd in zip(pred_answers, question_dicts)
    ]
    parsed_eval_results = [
        qd['choices'][pi][1] if pi is not None else 0
        for pi, qd in zip(pred_indices, question_dicts)
    ]

    parsed_eval_results_positives = []
    parsed_eval_results_negatives = []
    details_positives = []
    details_negatives = []
    details_recognition = []
    recognition_result = None

    for q_idx, (parsed_eval_result, question_dict) in enumerate(zip(parsed_eval_results, question_dicts)):
        detail_item = {
            **question_dict,
            'pred_answer': pred_answers[q_idx],
            'pred_index': pred_indices[q_idx],
            'eval_result': parsed_eval_result,
        }

        if question_dict['type'] == 'recognition':
            if parsed_eval_result == "correct":
                recognition_result = True
            elif parsed_eval_result == "incorrect":
                recognition_result = False
                print(f"Recognition is incorrect for key {key}, setting score to at most 0 for all questions")
            else:
                raise ValueError(f"Invalid recognition result: {parsed_eval_result}")
            details_recognition.append(detail_item)
        elif question_dict['type'] == 'negative':
            if recognition_result is False:
                parsed_eval_result = min(0, parsed_eval_result)
            parsed_eval_results_negatives.append(parsed_eval_result)
            details_negatives.append(detail_item)
        elif question_dict['type'] == 'positive':
            if recognition_result is False:
                parsed_eval_result = min(0, parsed_eval_result)
            parsed_eval_results_positives.append(parsed_eval_result)
            details_positives.append(detail_item)

    score_pos = sum(parsed_eval_results_positives) / len(parsed_eval_results_positives) if parsed_eval_results_positives else 0
    score_neg = sum(parsed_eval_results_negatives) / len(parsed_eval_results_negatives) if parsed_eval_results_negatives else None
    denom = len(parsed_eval_results_positives) + (len(parsed_eval_results_negatives) if parsed_eval_results_negatives else 0)
    score = (sum(parsed_eval_results_positives) + sum(parsed_eval_results_negatives)) / denom if denom > 0 else 0

    return dict(
        details_positives=details_positives,
        details_negatives=details_negatives,
        details_recognition=details_recognition,
        prompt=prompt,
        response=response,
        score=score,
        score_pos=score_pos,
        score_neg=score_neg,
        recognition_result=recognition_result,
    )


def mask_to_box(mask_np):
    mask_coords = np.argwhere(mask_np)
    #print(mask_coords)
    # if len(mask_coords) == 0:
    #     return 0, 0, 0, 0
    y0, x0 = mask_coords.min(axis=0)
    y1, x1 = mask_coords.max(axis=0) + 1
    return x0, y0, (x1 - x0), (y1 - y0)


def encode_pil_image_to_base64(pil_image):
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def load_anno_data(anno_json_path: str) -> dict:
    """Return {image_id: {file_name, height, width, annotations: {ann_index: ann}}}."""
    with open(anno_json_path) as f:
        data = json.load(f)
    lookup = {}
    for item in data:
        image_id = str(item["image_id"])
        ann_by_index = {}
        for ann in item["annotations"]:
            ann_by_index[str(ann["index"])] = ann
        lookup[image_id] = {
            "file_name": item["file_name"],
            "height": item["height"],
            "width": item["width"],
            "annotations": ann_by_index,
        }
    return lookup


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate model outputs (anno format)')
    parser.add_argument("--pred", type=str, required=True,
                        help="Path to the prediction JSON file (keyed by image_id_annIndex)")
    parser.add_argument('--qa', type=str, default="annotations/qa.json",
                        help='Path to the reference QA file')
    parser.add_argument('--class-names', type=str, default="annotations/class_names.json",
                        help='Path to the class names JSON file')
    parser.add_argument('--anno-json', type=str, default="annotations/annotations.json",
                        help='Path to anno JSON file with RLE masks')
    parser.add_argument('--image-root', type=str, default="annotations/images",
                        help='Root directory for images')
    parser.add_argument('--api-call-limit', type=int, default=5000, help='API call limit')
    parser.add_argument('--suffix', type=str, default="", help='Suffix for the evaluation file')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--concurrency', type=int, default=DEFAULT_CONCURRENCY)

    args = parser.parse_args()

    eval_file = os.path.splitext(args.pred)[0] + f"_eval_gpt{args.suffix}.json"

    # Load existing eval results for caching
    eval_results = {}
    if os.path.exists(eval_file):
        with open(eval_file) as f:
            eval_results = json.load(f)
    
    print(f"Loaded {len(eval_results)} existing eval results from {eval_file} for caching.")

    with open(args.pred) as f:
        data_pred = json.load(f)
        
    print(f"Loaded {len(data_pred)} predictions from {args.pred}")

    with open(args.qa) as f:
        data_qa = json.load(f)

    with open(args.class_names) as f:
        data_class_names = json.load(f)

    # Build anno lookup for image/mask data
    print(f"Loading anno data from {args.anno_json}...")
    anno_lookup = load_anno_data(args.anno_json)
    print(f"Loaded {len(anno_lookup)} images from anno JSON.")

    p = inflect.engine()

    scores = {}
    scores_pos = {}
    scores_neg = {}
    missing_key_count = 0

    keys = list(data_qa.keys())
    
    print(f"Evaluating {len(keys)} samples with concurrency={args.concurrency}...")

    # Pre-cache loaded images to avoid re-loading for same image_id
    _image_cache = {}

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        for key in tqdm(keys, disable=args.quiet):
            key = str(key)

            # Parse key → image_id, ann_index
            parts = key.rsplit("_", 1)
            if len(parts) != 2:
                print(f"[WARN] Skipping malformed key: {key}")
                continue
            image_id, ann_index = parts
            #print(f"Processing key: {key} (image_id: {image_id}, ann_index: {ann_index})")
            if image_id not in anno_lookup:
                print(f"[WARN] image_id {image_id} not found in anno data, skipping {key}")
                continue
            img_data = anno_lookup[image_id]
            if ann_index not in img_data["annotations"]:
                print(f"[WARN] ann_index {ann_index} not found for image {image_id}, skipping {key}")
                continue

            ann = img_data["annotations"][ann_index]
            img_h, img_w = img_data["height"], img_data["width"]
            file_name = img_data["file_name"]
            file_name = os.path.join(args.image_root, file_name)

            # Load image (with caching per image_id)
            if image_id not in _image_cache:
                if not os.path.exists(file_name):
                    print(f"[WARN] Image not found: {file_name}, skipping {key}")
                    continue
                try:
                    _image_cache[image_id] = Image.open(file_name).convert("RGB")
                except Exception as e:
                    print(f"[ERROR] Loading {file_name}: {e}, skipping {key}")
                    continue
            img = _image_cache[image_id]

            # Decode RLE mask
            mask_np = decode_rle(ann["mask_rle"], img_h, img_w).astype(bool)

            img_np = np.array(img)
            assert img_np.shape[:2] == mask_np.shape, \
                f"image shape {img_np.shape} mismatches mask shape {mask_np.shape} for {key}"

            pil_mask = Image.fromarray((mask_np * 255).astype(np.uint8))
            #print(mask_np.sum())
            # Focal crop (same logic as original eval script)
            x0, y0, w, h = mask_to_box(mask_np)
            xc, yc = x0 + w / 2, y0 + h / 2
            w, h = max(w, 56), max(h, 56)
            x0, y0 = int(xc - w / 2), int(yc - h / 2)

            cropped_img_np = img_np[
                max(y0 - h, 0):min(y0 + 2 * h, img_h),
                max(x0 - w, 0):min(x0 + 2 * w, img_w),
            ]
            cropped_mask_np = mask_np[
                max(y0 - h, 0):min(y0 + 2 * h, img_h),
                max(x0 - w, 0):min(x0 + 2 * w, img_w),
            ]

            cropped_pil_img = Image.fromarray(cropped_img_np)
            cropped_pil_mask = Image.fromarray((cropped_mask_np * 255).astype(np.uint8))

            images = [
                encode_pil_image_to_base64(cropped_pil_img),
                encode_pil_image_to_base64(cropped_pil_mask),
            ]

            # Use cached GPT responses if available
            if key in eval_results and isinstance(eval_results[key], dict):
                response_override = eval_results[key].get('response')
            else:
                response_override = None

            # Get prediction
            if key not in data_pred:
                print(f"[WARN] Key {key} not found in prediction data, skipping")
                print(len(data_pred), len(data_qa))
                missing_key_count += 1
                continue
            pred_value = data_pred[key]

            # Build recognition question from class_names
            class_name = data_class_names[key]
            recognition_question_dict = {
                "question": (
                    f"The object in the image is {class_name}. Based on the image, "
                    f"is it likely that the object in the description is given class: "
                    f"{class_name} or object of a similar type?"
                ),
                "choices": [("Yes", "correct"), ("No", "incorrect")],
                "type": "recognition",
            }

            question_dicts = [recognition_question_dict, *data_qa[key]]
            info = evaluate(
                question_dicts=question_dicts,
                pred_caption=pred_value,
                images=images,
                temperature=0.,
                max_tokens=300,
                response_override=response_override,
                key=key,
                executor=executor,
            )

            scores[key] = info["score"]
            scores_pos[key] = info["score_pos"]
            scores_neg[key] = info["score_neg"]
            eval_results[key] = {"pred": pred_value, **info}

    # Compute averages
    avg_score_pos = sum(scores_pos.values()) / len(scores_pos) if scores_pos else 0
    valid_neg = [v for v in scores_neg.values() if v is not None]
    avg_score_neg = sum(valid_neg) / len(valid_neg) if valid_neg else 0
    
    eval_results["avg_pos"] = avg_score_pos
    eval_results["avg_neg"] = avg_score_neg

    with open(eval_file, "w") as f:
        json.dump(eval_results, f, indent=4)

    print(f"Missing keys: {missing_key_count}")
    print(f"Average Positive Score: {avg_score_pos:.3f}")
    print(f"Average Negative Score: {avg_score_neg:.3f}")
    print(f"Summary (Pos\tNeg\tAvg(Pos, Neg)):\t{avg_score_pos:.3f},\t{avg_score_neg:.3f},\t{(avg_score_pos + avg_score_neg) / 2:.3f}")
    print(f"Evaluation data saved to {eval_file}")

"""
python eval_gpt_with_image.py \
    --pred parallel_model_outputs_cache/paradlc_outputs.json \
    --qa annotations/qa.json \
    --class-names annotations/class_names.json
    --anno-json annotations/annotations.json
"""