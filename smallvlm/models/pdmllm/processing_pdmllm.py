
import math
import torch
import warnings
import PIL.Image

from torch.nn import functional as F
from collections import UserDict, OrderedDict
from typing import Union, Optional, Tuple, List, Dict, Any

from transformers.image_utils import load_image
from transformers.feature_extraction_utils import BatchFeature
from .chat_template_utils import render_jinja_template
from transformers.processing_utils import ProcessorMixin, AllKwargsForChatTemplate


class PDMLLMProcessor(ProcessorMixin):
    attributes = ["tokenizer", "image_processor"]
    optional_attributes = ['chat_template']
    model_input_names = ['input_ids', 'attention_mask', 'pixel_values']
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
            self, tokenizer, image_processor, chat_template=None,
            image_size=512,
            patch_size=16,
            downsample_ratio=0.5,
            max_sub_img=6,
            min_sub_img=1,
            image_token='<IMG_CONTEXT>',
            image_start_token='<img>',
            image_end_token='</img>',
            special_tokens=['<IMG_CONTEXT>', '<img>', '</img>'],
            **kwargs):
        if chat_template is None:
            chat_template = "{% for message in messages %}{% if loop.first and message['role'] != 'system' %}<|start_header_id|>system<|end_header_id|>\nYou are a helpful assistant.<|eot_id|>\n{% endif %}<|start_header_id|>{{ message['role'] }}<|end_header_id|>\n{% if message['role'] == 'assistant' %}{% generation %}{{ message['content'][0]['text'] }}<|eot_id|>{% endgeneration %}{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}<img><IMG_CONTEXT></img>{% elif content['type'] == 'video' or 'video' in content %}<video><VIDEO_CONTEXT></video>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|eot_id|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\n{% endif %}"
        super().__init__(tokenizer=tokenizer, image_processor=image_processor, chat_template=chat_template)
        if isinstance(image_size, List) or isinstance(image_size, Tuple):
            image_size = image_size[0]
        self.num_image_token = int((image_size // patch_size) ** 2 * (downsample_ratio ** 2))

        self.vision_token_share_pe = kwargs.get('vision_token_share_pe', True)
        self.image_token_len = kwargs.pop('image_token_len', 256)
        self.max_sub_img = max_sub_img
        self.min_sub_img = min_sub_img

        self.image_token = image_token
        self.image_start_token = image_start_token
        self.image_end_token = image_end_token
        special_tokens = special_tokens + [f'<|Mask_Cap_{i}|>' for i in range(16)]
        self.tokenizer.add_special_tokens({'additional_special_tokens': special_tokens}, replace_additional_special_tokens=False)
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)
        self.image_start_token_id = self.tokenizer.convert_tokens_to_ids(self.image_start_token)
        self.image_end_token_id = self.tokenizer.convert_tokens_to_ids(self.image_end_token)
        if 'llada' in tokenizer.name_or_path.lower():
            self._pad_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")

        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        else:
            image_size = image_size
        self.image_size = image_size
        assert image_size[0] == image_size[1]

    def apply_chat_template(self, conversation, chat_template = None, **kwargs) -> str:
        if chat_template is None:
            chat_template = self.chat_template

        # Split template kwargs from processor/tokenization kwargs so that
        # `tokenize=True` can reuse the processor pipeline without polluting
        # the template rendering inputs.
        tokenize = kwargs.pop("tokenize", False)
        return_dict = kwargs.pop("return_dict", False)
        return_tensors = kwargs.pop("return_tensors", None)
        images = kwargs.pop("images", [])
        videos = kwargs.pop("videos", None)

        if not images:
            for message in conversation:
                content = message.get("content", [])
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and (item.get("type") == "image" or "image" in item):
                            image = item.get("image") or item.get("image_url")
                            if image is not None:
                                images.append(image)

        processor_kwargs = {}
        for key in ("padding", "truncation", "max_length"):
            if key in kwargs:
                processor_kwargs[key] = kwargs.pop(key)
        if return_tensors is not None:
            processor_kwargs["return_tensors"] = return_tensors

        processed_kwargs = {
            "mm_load_kwargs": {},
            "template_kwargs": {},
        }
        # for kwarg_type in processed_kwargs:
        #     for key in AllKwargsForChatTemplate.__annotations__[kwarg_type].__annotations__.keys():
        #         kwarg_type_defaults = AllKwargsForChatTemplate.__annotations__[kwarg_type]
        #         default_value = getattr(kwarg_type_defaults, key, None)
        #         value = kwargs.pop(key, default_value)
        #         if value is not None and not isinstance(value, dict):
        #             processed_kwargs[kwarg_type][key] = value

        # Pass unprocessed custom kwargs
        processed_kwargs["template_kwargs"].update(kwargs)
        conversations = [conversation]

        prompt, generation_indices = render_jinja_template(
            conversations=conversations,
            chat_template=chat_template,
            return_assistant_tokens_mask=True,
            **processed_kwargs["template_kwargs"],  # different flags such as `return_assistant_mask`
            **self.tokenizer.special_tokens_map,  # tokenizer special tokens are used by some templates
        )

        if not tokenize:
            return prompt, generation_indices

        # Reuse the processor pipeline to produce tokenized inputs.
        model_inputs = self(
            text=prompt,
            images=images,
            videos=videos,
            generation_indices=generation_indices,
            **processor_kwargs,
        )
        # if return_dict:
        #     return model_inputs
        return model_inputs

    def __call__(self, text=None, images=[], videos=None, generation_indices=None, **kwargs) ->BatchFeature:
        inputs = self.tokenizer(text, padding=False, truncation=False, return_attention_mask=False)
        assistant_masks = []
        input_ids = inputs["input_ids"]
        for i in range(len(input_ids)):
            current_mask = [0] * len(input_ids[i])
            if 'llada' in self.tokenizer.name_or_path.lower():
                for assistant_start_char, assistant_end_char in generation_indices[i]:
                    start_token = inputs.char_to_token(i, assistant_start_char)
                    end_token = inputs.char_to_token(i, assistant_end_char - 1)
                    if start_token is None:
                        # start_token is out of bounds maybe due to truncation.
                        break
                    for token_id in range(start_token, end_token + 1 if end_token else len(input_ids[i])):
                        current_mask[token_id] = 1
            
            assistant_masks.append(current_mask)

        inputs["assistant_masks"] = assistant_masks[0]
        inputs['input_ids'] = input_ids[0]

        truncation = kwargs.pop('truncation', False)
        max_length = kwargs.pop('max_length', 1024)
        padding = kwargs.pop('padding', False)

        inputs = self.process_images(images, inputs=inputs)
        if isinstance(inputs, UserDict):
            inputs = inputs.data
        
        if 'attention_mask' not in inputs:
            inputs['attention_mask'] = [1] * len(inputs['input_ids'])
        if 'assistant_masks' in inputs:
            inputs['prompt_mask'] = [1-x for x in inputs.pop('assistant_masks')]

        inputs = self.process_inputs(inputs)
        if truncation and len(inputs['input_ids']) > max_length:
            inputs = self.truncate(inputs, max_length)
        if padding and len(inputs['input_ids']) < max_length:
            inputs = self.padding(inputs, max_length)

        inputs = self.to_tensor(inputs)
        self.check(inputs)
        if self.vision_token_share_pe:
            position_ids = self.get_position_ids(inputs)
            position_ids = torch.tensor([position_ids], dtype=torch.long)
            inputs['position_ids'] = position_ids

        inputs.pop('sub_image_nums', None)

        return BatchFeature(inputs)

    def get_position_ids(self, inputs: Dict[str, Any]):
        input_ids = inputs['input_ids'][0]
        image_token_lens = self.get_image_token_length(inputs)
        position_ids = []
        i, j = 0, 0
        while len(position_ids) < len(input_ids):
            if input_ids[len(position_ids)] == self.image_token_id:
                image_token_len = image_token_lens[j]
                assert image_token_len % self.image_token_len == 0
                num_views = image_token_len // self.image_token_len
                for _ in range(num_views):
                    position_ids += [i] * self.image_token_len # 同一个图像的所有 token 共享相同的位置编码
                    i += 1
                j += 1
            else:
                position_ids.append(i)
                i += 1

        assert j == len(image_token_lens) and len(position_ids) == len(input_ids), \
            f"Wrong position_ids, {j} != {len(image_token_lens)} or {len(position_ids)} != {len(input_ids)}"

        return position_ids
    
    def process_images(self, images, inputs):
        images = [load_image(img) for img in images]
        if len(images) > 0:
            processed_images = []
            sub_image_nums = []
            for image in images:
                if len(images) > 1:
                    # for multi images, remove the split strategy
                    sub_images = dynamic_preprocess(
                        image, min_num=1,
                        max_num=1,
                        image_size=self.image_size[0], use_thumbnail=True)
                else:
                    sub_images = dynamic_preprocess(
                        image, min_num=self.min_sub_img,
                        max_num=self.max_sub_img,
                        image_size=self.image_size[0], use_thumbnail=True)

                sub_image_nums.append(len(sub_images))
                processed_images += sub_images
            # print([_img.size for _img in processed_images])
            pixel_values = self.image_processor.preprocess(
                images=processed_images, return_tensors="pt"
            )["pixel_values"] # (N, c, h, w)
        else:
            pixel_values = torch.zeros((
                1, 3, self.image_size[0], self.image_size[1]), dtype=torch.float32
            )
            sub_image_nums = []

        inputs['pixel_values'] = pixel_values
        inputs['sub_image_nums'] = sub_image_nums
        return inputs
    
    def truncate(self, inputs: Dict[str, Any], max_length: int):
        assert self.image_token_id not in inputs['input_ids'][max_length:], f"Truncate image token is not allowed."
        inputs['input_ids'] = inputs['input_ids'][:max_length]
        inputs['attention_mask'] = inputs['attention_mask'][:max_length]
        if 'prompt_mask' in inputs:
            inputs['prompt_mask'] = inputs['prompt_mask'][:max_length]
        return inputs

    def get_image_token_length(self, inputs: Dict[str, Any]) -> List[int]:
        sub_image_nums = inputs.get('sub_image_nums', None)
        if sub_image_nums is None or len(sub_image_nums) == 0:
            return []
        image_token_lens = [_num * self.num_image_token for _num in sub_image_nums]
        return image_token_lens

    def process_inputs(self, inputs: Dict[str, Any]):
        graft_token_lens = self._get_graft_token_length(inputs)
        inputs['input_ids'] = self._graft_token(inputs['input_ids'], graft_token_lens, self.image_token_id)
        inputs['attention_mask'] = self._graft_token(inputs['attention_mask'], graft_token_lens, 'replicate')
        if 'prompt_mask' in inputs:
            inputs['prompt_mask'] = self._graft_token(inputs['prompt_mask'], graft_token_lens, 'replicate')
        return inputs
    
    def _graft_token(self, seq, graft_token_lens, value):
        if value == 'replicate':
            for i in reversed(graft_token_lens.keys()):
                seq[i:] = [seq[i]] * graft_token_lens[i] + seq[i+1:]
        else:
            for i in reversed(graft_token_lens.keys()):
                seq[i:] = [value] * graft_token_lens[i] + seq[i+1:]
        return seq
    
    def _get_graft_token_length(self, inputs: Dict[str, Any]) -> Dict[int, int]:
        image_token_pos = [i for i, x in enumerate(inputs['input_ids']) if x == self.image_token_id]
        image_token_lens = self.get_image_token_length(inputs)
        assert len(image_token_pos) == len(image_token_lens), \
            "Wrong image token count, " \
            f"image_token_count({len(image_token_pos)}) != image_count({len(image_token_lens)})"

        graft_token_lens = OrderedDict(item for item in zip(image_token_pos, image_token_lens))
        return graft_token_lens
    
    def check(self, inputs: Dict[str, Any]):
        image_embed_token_count = torch.count_nonzero(inputs['input_ids'] == self.image_token_id).item()
        image_embed_count = sum(self.get_image_token_length(inputs))
        assert image_embed_token_count == image_embed_count, \
            "Wrong image embed token count, " \
            f"image_embed_token_count({image_embed_token_count}) != image_embed_count({image_embed_count})"

    def padding(self, inputs: Dict[str, Any], max_length: int):
        padding_len = max_length - len(inputs['input_ids'])
        inputs['input_ids'] += [self.pad_token_id] * padding_len
        inputs['attention_mask'] += [0] * padding_len
        if 'prompt_mask' in inputs:
            inputs['prompt_mask'] += [0] * padding_len
        return inputs
    
    def decode(self, token_ids: Union[List[int], torch.Tensor], **kwargs):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        text = self.tokenizer.decode(token_ids, **kwargs)
        return text

    def batch_decode(self, sequences: Union[List[List[int]], torch.Tensor], **kwargs):
        if isinstance(sequences, torch.Tensor):
            sequences = sequences.tolist()
        texts = self.tokenizer.batch_decode(sequences, **kwargs)
        return texts
    
    def to_tensor(self, inputs):
        inputs['input_ids'] = torch.tensor([inputs['input_ids']], dtype=torch.long)
        inputs['attention_mask'] = torch.tensor([inputs['attention_mask']], dtype=torch.bool)
        if 'prompt_mask' in inputs:
            inputs['prompt_mask'] = torch.tensor([inputs['prompt_mask']], dtype=torch.bool)
        return inputs
    
    @property
    def pad_token_id(self):
        return self._pad_token_id
    
    def __repr__(self):
        pass

    def __str__(self):
        return 'PDMLLMProcessor'

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    # print(f'width: {width}, height: {height}, best_ratio: {best_ratio}')
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=512, use_thumbnail=True):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images