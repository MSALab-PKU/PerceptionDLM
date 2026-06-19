import math
import pandas as pd
import random
import re
import yaml
import string
import torch
import torch.distributed as dist
import torchvision.transforms as T
import transformers
import warnings
from PIL import Image
from functools import partial
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoProcessor

from .utils import (build_multi_choice_prompt,
                    build_video_prompt,
                    build_mpo_prompt,
                    build_mcq_cot_prompt,
                    build_qa_cot_prompt,
                    mpo_post_processing,
                    format_nav_prompt,
                    pile_action_history,
                    reorganize_prompt,
                    load_image)
from .utils import mpo_prompt_with_final_answer, mpo_prompt_without_final_answer, parse_bbox_internvl

from ..base import BaseModel
from ...dataset import DATASET_TYPE, DATASET_MODALITY, build_dataset, infer_dataset_basename
from ...smp import *

R1_SYSTEM_PROMPT = """
You are an AI assistant that rigorously follows this response protocol:

1. First, conduct a detailed analysis of the question. Consider different \
angles, potential solutions, and reason through the problem step-by-step. \
Enclose this entire thinking process within <think> and </think> tags.

2. After the thinking section, provide a clear, concise, and direct answer to \
the user's question. Separate the answer from the think section with a newline.

Ensure that the thinking process is thorough but remains focused on the \
query. The final answer should be standalone and not reference the thinking \
section.
""".strip()


def prepare_messages_list(prompt, image_path, system_prompt=None):
    from lmdeploy.vl.constants import IMAGE_TOKEN
    content = [{'type': 'text', 'text': prompt.replace('<image>', IMAGE_TOKEN)}]

    if isinstance(image_path, str):
        image_path = [image_path]

    for image in image_path:
        img = Image.open(image).convert('RGB')
        b64 = encode_image_to_base64(img)
        img_struct = dict(url=f'data:image/jpeg;base64,{b64}')
        content.append(dict(type='image_url', image_url=img_struct))

    messages = []

    if system_prompt is not None:
        messages.append({'role': 'system', 'content': system_prompt})

    messages.append({
        'role': 'user',
        'content': content,
    })

    return [messages]


def extract_boxed_content(ans: str):
    idx = ans.rfind(r'\boxed{')
    if idx == -1:
        return ans

    idx += len(r'\boxed{')
    brace_level = 1
    content_start = idx
    i = idx

    while i < len(ans):
        if ans[i] == '{':
            brace_level += 1
        elif ans[i] == '}':
            brace_level -= 1
            if brace_level == 0:
                break
        i += 1

    if brace_level != 0:
        # Unbalanced braces
        return ans

    content = ans[content_start:i]
    return content


class DMLLMChat(BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True
    def __init__(self,
                 model_path,
                 load_in_8bit=False,
                 use_mpo_prompt=False,
                 version='V2.0',
                 screen_parse=True,
                 # Best-of-N parameters
                 best_of_n=1,
                 reward_model_path=None,
                 # R1 parameters
                 cot_prompt_version='v1',
                 #
                 use_lmdeploy=False,
                 use_postprocess=False,
                 **kwargs):

        assert best_of_n >= 1
        assert model_path is not None
        assert version_cmp(transformers.__version__, '4.37.2', 'ge')

        self.use_lmdeploy = use_lmdeploy
        self.cot_prompt_version = cot_prompt_version
        self.use_mpo_prompt = use_mpo_prompt
        self.use_cot = (os.getenv('USE_COT') == '1')
        self.use_postprocess = use_postprocess

        if cot_prompt_version == 'r1':
            self.system_prompt = R1_SYSTEM_PROMPT
            self.cot_prompt = 'Please answer the question and put the final answer within \\boxed{}.'
        elif cot_prompt_version == 'v2':
            self.system_prompt = None
            self.cot_prompt = "Answer the preceding multiple-choice question \
            by carefully analyzing the provided image. \nPlease answer with \
            carefully thought step by step. Apply the thinking process \
            recursively at both macro and micro levels. \nVerify consistency \
            of reasoning and look for potential flaws or gaps during \
            thinking. \nWhen realize mistakes, explain why the previous \
            thinking was incorrect, fix it and then continue thinking.\nThe \
            last line of your response should follow this format: 'Answer: \
            \\boxed{$LETTER}' (without quotes), where LETTER is one of the \
            options\n\n"
        else:
            assert cot_prompt_version == 'v1'
            self.system_prompt = None
            self.cot_prompt = None

        self.cot_prompt = "Please reason step by step, and answer the question using a single word or phrase."
        self.model_path = model_path
        
        # Regular expression to match the pattern 'Image' followed by a number, e.g. Image1
        self.pattern = r'Image(\d+)'
        # Replacement pattern to insert a hyphen between 'Image' and the number, e.g. Image-1
        self.replacement = r'Image-\1'

        # Convert InternVL2 response to dataset format
        # e.g. Image1 -> Image-1

        # Regular expression to match the pattern 'Image-' followed by a number
        self.reverse_pattern = r'Image-(\d+)'
        # Replacement pattern to remove the hyphen (Image-1 -> Image1)
        self.reverse_replacement = r'Image\1'

        self.screen_parse = screen_parse

        self.model = AutoModel.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
        ).eval().cuda()
        
        self.device = 'cuda'

        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.image_processor = self.processor.image_processor
        self.tokenizer = self.processor.tokenizer

        self.version = version

        self.kwargs = {
            "steps": 64,
            "gen_length": 64,
            "block_length": 64,
            "temperature": 0.0,
            "cfg_scale": 0.,
            "remasking": 'low_confidence',
        }
        
        self.kwargs.update(kwargs)
        warnings.warn(f'Following kwargs received: {self.kwargs}, will use as default generation config. ')

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        if dataset in [
            'atomic_dataset', 'electro_dataset', 'mechanics_dataset',
            'optics_dataset', 'quantum_dataset', 'statistics_dataset'
        ]:
            return False
        if listinstr(['MMDU', 'MME-RealWorld', 'MME-RealWorld-CN', 'WeMath_COT', 'MMAlignBench'], dataset):
            # For Multi-Turn we don't have custom prompt
            return False
        if DATASET_MODALITY(dataset) == 'VIDEO':
            # For Video benchmarks we don't have custom prompt at here
            return False
        else:
            return True

    def build_prompt(self, line, dataset=None):
        use_mpo_prompt = self.use_mpo_prompt and (self.use_cot or dataset in ['MMStar', 'HallusionBench', 'OCRBench'])

        assert self.use_custom_prompt(dataset)
        assert dataset is None or isinstance(dataset, str)
        # if dataset is not ChartMimic, dump image (assert "image_path" in line)
        if not listinstr(['ChartMimic'], dataset):
            tgt_path = self.dump_image(line, dataset)
        else:
            input_figure_path_rel = line["input_figure"]
            ROOT = LMUDataRoot()
            img_root = os.path.join(ROOT, 'images', 'ChartMimic')
            input_figure_path = os.path.join(img_root, input_figure_path_rel)
            tgt_path = [input_figure_path]

        if dataset is not None and DATASET_TYPE(dataset) == 'Y/N':
            question = line['question']
            if listinstr(['MME'], dataset):
                prompt = question
                # print(prompt)
                # prompt = question + ' Answer the question using a single word (Yes or No).'
                # prompt = question + ' Answer the question using Yes or No.'
            elif listinstr(['HallusionBench', 'AMBER'], dataset):
                prompt = question + ' Please answer yes or no. Answer the question using a single word or phrase.'
            else:
                prompt = question
        elif dataset is not None and DATASET_TYPE(dataset) == 'MCQ':
            prompt = build_multi_choice_prompt(line, dataset)
            
            if os.getenv('USE_COT') == '1':
                prompt = build_mcq_cot_prompt(line, prompt, self.cot_prompt)
        elif dataset is not None and DATASET_TYPE(dataset) == 'VQA':
            question = line['question']
            if listinstr(['LLaVABench', 'WildVision'], dataset):
                prompt = question + '\nAnswer this question in detail.'
            elif listinstr(['OCRVQA', 'TextVQA', 'ChartQA', 'DocVQA', 'InfoVQA', 'OCRBench',
                            'DUDE', 'SLIDEVQA', 'GQA', 'MMLongBench_DOC'], dataset):
                prompt = question + "\n\nPlease reason step by step, and answer the question using a single word or phrase."
            elif listinstr(['MathVista', 'MathVision', 'VCR', 'MTVQA', 'MMVet', 'MathVerse',
                            'MMDU', 'CRPE', 'MIA-Bench', 'MM-Math', 'DynaMath', 'QSpatial',
                            'WeMath', 'LogicVista', 'MM-IFEval', 'ChartMimic'], dataset):
                prompt = question
                self.cot_prompt = "\n\nPlease reason step by step."
                prompt = build_qa_cot_prompt(line, prompt, self.cot_prompt)
            else:
                prompt = question + '\nAnswer the question using a single word or phrase.'
        elif dataset is not None and DATASET_TYPE(dataset) == 'GUI':
            ds_basename = infer_dataset_basename(dataset)
            ds = build_dataset(dataset, skeleton=True)
            action_space = ds.get_action_space()
            traj_dict = ds.get_trajectory(line)

            prompt_config = GUI_TEMPLATE[ds_basename]
            if 'history' in prompt_config["placeholders"]:
                traj_dict['history'] = pile_action_history(traj_dict['history'])
            prompt = format_nav_prompt(
                (
                    "Please provide the bounding box coordinate of the region this sentence describes: <ref>{task}</ref>"  # noqa: E501
                    if self.screen_parse
                    else prompt_config["template"]
                ),
                prompt_config["placeholders"],
                action_space=action_space,
                **traj_dict,
            )
        else:
            # VQA_ex_prompt: OlympiadBench, VizWiz
            prompt = line['question']
            if os.getenv('USE_COT') == '1':
                prompt = build_qa_cot_prompt(line, prompt, self.cot_prompt)

        message = [dict(type='text', value=prompt)]
        message.extend([dict(type='image', value=s) for s in tgt_path])

        if use_mpo_prompt:
            message = build_mpo_prompt(message, line, dataset)
        return message

    def get_generation_config(self, dataset):
        kwargs = self.kwargs.copy()
        if dataset is None:
            return kwargs
            
        dataset_lower = dataset.lower()
        if any(x in dataset_lower for x in ['mathvista_mini', 'mathverse_mini_vision_only']):
            kwargs.update({"steps": 256, "gen_length": 256, "block_length": 1})
        elif any(x in dataset_lower for x in ['chartqa_test', 'docvqa_val', 'infovqa_val']):
            kwargs.update({"steps": 128, "gen_length": 128, "block_length": 128})
        elif any(x in dataset_lower for x in ['mmstar', 'seedbench_img', 'mmbench_dev_en', 'mmmu_dev_val', 'ai2d', 'mmvp', 'blink', 'realworldqa', 'cv-bench-2d', 'hallusionbench', 'vstarbench']):
            kwargs.update({"steps": 2, "gen_length": 2, "block_length": 2})
            
        return kwargs

    def set_max_num(self, dataset):
        # The total limit on the number of images processed, set to avoid Out-of-Memory issues.
        self.total_max_num = 64
        if dataset is None:
            self.max_num = 6
            return None
        res_12_datasets = ['ChartQA_TEST', 'MMMU_DEV_VAL', 'MMMU_TEST', 'MME-RealWorld',
                           'VCR_EN', 'VCR_ZH', 'OCRVQA']
        res_18_datasets = ['DocVQA_VAL', 'DocVQA_TEST', 'DUDE', 'MMLongBench_DOC', 'SLIDEVQA']
        res_24_datasets = ['InfoVQA_VAL', 'InfoVQA_TEST', 'OCRBench', 'HRBench4K', 'HRBench8K']
        if DATASET_MODALITY(dataset) == 'VIDEO':
            self.max_num = 1
        else:
            self.max_num = 6

    @torch.no_grad()
    def generate_dmllm(self, message, dataset=None):
        image_num = len([x for x in message if x['type'] == 'image'])
        max_num = max(1, min(self.max_num, self.total_max_num // image_num))
        prompt = reorganize_prompt(message, image_num, dataset=dataset)

        if dataset is not None and DATASET_MODALITY(dataset) == 'VIDEO':
            prompt = build_video_prompt(prompt, dataset)

        if image_num > 1:
            image_path = [x['value'] for x in message if x['type'] == 'image']
            num_patches_list, pixel_values_list = [], []
            for image_idx, file_name in enumerate(image_path):
                upscale_flag = image_idx == 0 and dataset is not None and listinstr(['MMMU'], dataset)
                curr_pixel_values = load_image(
                    file_name, max_num=max_num, upscale=upscale_flag, processor=self.image_processor
                ).to(self.device).to(torch.bfloat16)
                num_patches_list.append(curr_pixel_values.size(0))
                pixel_values_list.append(curr_pixel_values)
            pixel_values = torch.cat(pixel_values_list, dim=0)
        elif image_num == 1:
            image_path = [x['value'] for x in message if x['type'] == 'image'][0]
            upscale_flag = dataset is not None and listinstr(['MMMU'], dataset)
            pixel_values = load_image(
                image_path, max_num=max_num, upscale=upscale_flag,
                processor=self.image_processor
            ).to(self.device).to(torch.bfloat16)
            num_patches_list = [pixel_values.size(0)]
        else:
            pixel_values = None
            num_patches_list = []

        generation_config = self.get_generation_config(dataset)
        response = self.model.chat(
            self.tokenizer,
            pixel_values=pixel_values,
            num_patches_list=num_patches_list,
            question=prompt,
            generation_config=generation_config,
            verbose=True,
        )


        response = response.split("<|eot_id|>")[0].strip()
        response = response.replace("<|im_end|>", "")
        if ("Yes" in response or "No" in response) and len(response) <= 4:
            if "Yes" in response:
                response = "Yes"
            else:
                response = "No"
        elif len(response) > 2 and response[1] == "." and response[0] in "ABCDEF":
            response = response[0]
        # print("Processed response: ", response)
        return response

    def generate_inner(self, message, dataset=None):
        self.set_max_num(dataset)
        return self.generate_dmllm(message, dataset)