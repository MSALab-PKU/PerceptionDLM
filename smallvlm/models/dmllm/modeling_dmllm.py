from typing import Optional, List
import torch
from torch import nn
from torch.nn import functional as F
import transformers
from transformers import PreTrainedModel, AutoModel, AutoModelForCausalLM, GenerationConfig
from transformers import AutoConfig
from transformers.feature_extraction_utils import BatchFeature
from .configuration_dmllm import DMLLMConfig
from .modeling_abstractor import PerceiverProjection
from .modeling_llada import LLaDAModelLM
from .cache import *
from .configuration_llada import LLaDAConfig

def build_vision_model(config, model=None):
    assert hasattr(config, "name_or_path")
    if model is None:
        model = AutoModel.from_pretrained(
            config.name_or_path, config=config, trust_remote_code=True)
    return model

class DMLLM(PreTrainedModel):
    config_class = DMLLMConfig
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = "past_key_values"
    _supports_cache_class = False
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    accepts_loss_kwargs=False

    def __init__(self,
                 config: DMLLMConfig,
                 language_model=None,
                 vision_model=None,
                 processor=None):
        super().__init__(config)
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.downsample_ratio = config.downsample_ratio
        self.num_image_token = config.num_image_token
        self.vision_select_layer = config.vision_select_layer
        self.replacement_noise_mode = config.replacement_noise_mode

        try:
            vision_hidden_states = self.config.vision_model_config.hidden_size
        except:
            vision_hidden_states = self.config.vision_model_config.vision_config.hidden_size
            self.config.vision_model_config.hidden_size = vision_hidden_states

        vision_model = build_vision_model(config.vision_model_config, vision_model)

        vision_abstractor = PerceiverProjection(**config.vision_abstractor_config,
                                                in_dim=self.config.vision_model_config.hidden_size * (int(1 / self.downsample_ratio) ** 2),
                                                out_dim=self.config.language_model_config.hidden_size)

        if language_model is None:
            kwargs_ = {}
            if config._attn_implementation_internal is not None:
                kwargs_['attn_implementation'] = config._attn_implementation_internal
            if 'llada' in config.language_model_config.name_or_path.lower():
                with transformers.modeling_utils.no_init_weights():
                    language_model = LLaDAModelLM(config.language_model_config)
            else:
                raise ValueError(f"Unsupported language model: {config.language_model_config.name_or_path}")

        self.vision_model = vision_model
        self.vision_abstractor = vision_abstractor
        self.language_model = language_model

    def forward_vision(self, pixel_values):
        # pixel_values: (n, c, h, w) or (b, tiles, c, h, w)

        # Handle BatchFeature input
        if isinstance(pixel_values, BatchFeature):
            pixel_values = pixel_values["pixel_values"]
        
        # Handle 5D input: (b, tiles, c, h, w) -> (b*tiles, c, h, w)
        if pixel_values.dim() == 5:
            b, tiles, c, h, w = pixel_values.shape
            pixel_values = pixel_values.view(b * tiles, c, h, w)
        
        # flags for dummy images (all-zero images)
        image_flags = torch.sum(pixel_values, dim=(1, 2, 3)) != 0
        image_flags = image_flags.long()
        if image_flags.dim() > 1:
            image_flags = image_flags.squeeze(-1)

        # extract vision features
        if self.vision_select_layer == -1:
            image_embeddings = self.vision_model.vision_model(
                pixel_values=pixel_values,
            ).last_hidden_state
        else:
            image_embeddings = self.vision_model.vision_model(
                pixel_values=pixel_values, output_hidden_states=True
            ).hidden_states[self.vision_select_layer] # (B, N, C)
        vit_embeds = image_embeddings[image_flags == 1]

        if self.downsample_ratio != 1:
            patch_num = self.image_size // self.patch_size
            vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], patch_num, patch_num, vit_embeds.shape[-1])
            vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
            vit_embeds = vit_embeds.flatten(1, 2)

        vit_embeds = self.vision_abstractor(vit_embeds)

        return vit_embeds

    def prepare_for_lm(self, input_ids, vision_embeds):
        inputs_embeds = self.get_input_embeddings()(input_ids)
        vision_embeds_ = vision_embeds
        if vision_embeds is not None:
            try:
                vision_mask = input_ids == self.config.image_token_id
                if torch.count_nonzero(vision_mask).item() != vision_embeds.shape[:-1].numel():
                    info = "vision embeddings mismatch input embeddings: " \
                           f"vision_mask shape={vision_mask.shape}; " \
                           f"vision_mask count={torch.count_nonzero(vision_mask)}; " \
                           f"vision_embeds shape={vision_embeds.shape}"
                    #print(info)
                    num_vision_1 = torch.count_nonzero(vision_mask).item()
                    num_vision_2 = vision_embeds.shape[:-1].numel()
                    vision_embeds = vision_embeds.contiguous()
                    if num_vision_1 <= num_vision_2:
                        vision_embeds = vision_embeds.view(-1, vision_embeds.size(-1))[:num_vision_1]
                    else:
                        vision_embeds = vision_embeds.view(-1, vision_embeds.size(-1))
                        less_nums = num_vision_1 - num_vision_2
                        vision_embeds = torch.cat([vision_embeds, vision_embeds[-less_nums:]], dim=0)
                    vision_embeds = vision_embeds.contiguous()

                # assert torch.count_nonzero(vision_mask).item() == vision_embeds.shape[:-1].numel(), \
                #     "vision embeddings mismatch input embeddings: " \
                #     f"vision_mask shape={vision_mask.shape}; " \
                #     f"vision_mask count={torch.count_nonzero(vision_mask)}; " \
                #     f"vision_embeds shape={vision_embeds.shape}"
                inputs_embeds = torch.masked_scatter(inputs_embeds, vision_mask.unsqueeze(-1),
                                                     vision_embeds.to(inputs_embeds.dtype).view(-1,
                                                                                                vision_embeds.size(-1)))
            except:
                inputs_embeds = inputs_embeds + torch.sum(vision_embeds_[0, 0, :]) * 0.0

        return inputs_embeds

    def pixel_shuffle(self, x, scale_factor=0.5):
        x = x.contiguous()
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))
        x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def forward(self,
                input_ids: torch.LongTensor = None,
                attention_mask: Optional[torch.BoolTensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                pixel_values: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                labels: Optional[torch.LongTensor] = None,
                return_dict: bool = True,
                **kwargs,
                ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # ========Get visual embedding========
        if pixel_values is not None:
            vision_embeds = self.forward_vision(pixel_values)
        else:
            vision_embeds = None
        
        # print(f"input_ids.shape: {input_ids.shape}", {vision_embeds.shape})
        inputs_embeds = self.prepare_for_lm(input_ids, vision_embeds)
        # print(f"inputs_embeds.shape: {inputs_embeds.shape}")
        p_mask = None
        answer_length = None

        if self.is_gradient_checkpointing and torch.is_grad_enabled():
            inputs_embeds.requires_grad_(True)
        # ========Forward into LM========
        outputs = self.language_model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            return_dict=return_dict,
            labels=labels,
            use_cache=False,
            conversation_ids=None,
            replacement_noise_mode=self.replacement_noise_mode,
            p_mask = p_mask,
            answer_length = answer_length,
            **kwargs,
        )

        return outputs

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        super().gradient_checkpointing_enable(gradient_checkpointing_kwargs)
        self.language_model.enable_input_require_grads()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    def get_decoder(self):
        return self.language_model.get_decoder()

    def tie_weights(self):
        return self.language_model.tie_weights()

    @torch.no_grad()
    def generate(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.FloatTensor] = None,
            **generate_kwargs,
    ) -> torch.LongTensor:
        if pixel_values is not None:
            vision_embeds = self.forward_vision(pixel_values)
        else:
            vision_embeds = None
        
        inputs_embeds = self.prepare_for_lm(input_ids, vision_embeds)

        if 'llada' in self.config.language_model_config.name_or_path.lower():
            outputs = self.language_model.generate_with_embeds(
                inputs_embeds=inputs_embeds, **generate_kwargs
            )
        else:
            raise NotImplementedError(f"Generation not implemented for model: {self.config.language_model_config.name_or_path}")
        return outputs

    @torch.no_grad()
    def generate_replace_noise(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.FloatTensor] = None,
            **generate_kwargs,
    ) -> torch.LongTensor:
        if pixel_values is not None:
            vision_embeds = self.forward_vision(pixel_values)
        else:
            vision_embeds = None

        inputs_embeds = self.prepare_for_lm(input_ids, vision_embeds)

        outputs, all_steps_response = self.language_model.generate_with_embeds_replace_noise(
            inputs_embeds=inputs_embeds, **generate_kwargs
        )
        return outputs, all_steps_response

    def get_template(self):
        template = dict(
            SYSTEM=("<|start_header_id|>system<|end_header_id|>\n{system}<|eot_id|>\n"),
            INSTRUCTION=("<|start_header_id|>user<|end_header_id|>\n{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"),
            SUFFIX="<|eot_id|>",
            SUFFIX_AS_EOS=True,
            SEP="\n",
            STOP_WORDS=["<|eot_id|>"],
        )
        return template

    @torch.no_grad()
    def chat(
            self,
            tokenizer,
            pixel_values,
            question,
            generation_config,
            history=None,
            return_history=False,
            num_patches_list=None,
            IMG_START_TOKEN='<img>',
            IMG_END_TOKEN='</img>',
            IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
            verbose=False

    ):
        if history is None and pixel_values is not None and '<image>' not in question:
            question = '<image>\n' + question

        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        template = self.get_template()
        eos_token_id = tokenizer.convert_tokens_to_ids(template["SUFFIX"].strip())

        history = "" if history is None else history
        prompt = history
        prompt = prompt + template["INSTRUCTION"].format(input=question)

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        prompt = prompt[::-1]
        for num_patches in num_patches_list[::-1]:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            prompt = prompt.replace('<image>'[::-1], image_tokens[::-1], 1)
        prompt = prompt[::-1]
        model_inputs = tokenizer(prompt, return_tensors='pt')
        device = torch.device(self.language_model.device if torch.cuda.is_available() else 'cpu')
        input_ids = model_inputs['input_ids'].to(device)
        attention_mask = model_inputs['attention_mask'].to(device)
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            **generation_config
        )
        # response = [
        #     tokenizer.decode(g[len(p) :].tolist())
        #     for p, g in zip(input_ids, generation_output)
        # ][0]
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=False)[0]
        history = history + prompt + response
        response = response.split(template["SUFFIX"].strip())[0].strip()
        if return_history:
            return response, history
        else:
            if verbose:
                print(response)
            return response
        return

    @torch.no_grad()
    def chat_replace_noise(
            self,
            tokenizer,
            pixel_values,
            question,
            generation_config,
            history=None,
            return_history=False,
            num_patches_list=None,
            IMG_START_TOKEN='<img>',
            IMG_END_TOKEN='</img>',
            IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
            verbose=False

    ):
        if history is None and pixel_values is not None and '<image>' not in question:
            question = '<image>\n' + question

        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        template = self.get_template()
        eos_token_id = tokenizer.convert_tokens_to_ids(template["SUFFIX"].strip())

        history = "" if history is None else history
        prompt = history
        prompt = prompt + template["INSTRUCTION"].format(input=question)

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        prompt = prompt[::-1]
        for num_patches in num_patches_list[::-1]:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            prompt = prompt.replace('<image>'[::-1], image_tokens[::-1], 1)
        prompt = prompt[::-1]
        model_inputs = tokenizer(prompt, return_tensors='pt')
        device = torch.device(self.language_model.device if torch.cuda.is_available() else 'cpu')
        input_ids = model_inputs['input_ids'].to(device)
        attention_mask = model_inputs['attention_mask'].to(device)
        generation_config['eos_token_id'] = eos_token_id
        generation_output, all_steps_response = self.generate_replace_noise(
            pixel_values=pixel_values,
            input_ids=input_ids,
            **generation_config
        )
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=False)[0]

        all_steps_response_ = []
        for step_response in all_steps_response:
            step_response = tokenizer.batch_decode(step_response, skip_special_tokens=False)[0]
            all_steps_response_.append(step_response)
        all_steps_response = all_steps_response_
        for i, step_response in enumerate(all_steps_response):
            print(f"Step {i}: {step_response}\n")

        history = history + prompt + response
        response = response.split(template["SUFFIX"].strip())[0].strip()
        if return_history:
            return response, history
        else:
            if verbose:
                print(response)
            return response
        return

AutoConfig.register("dmllm", DMLLMConfig)
AutoModel.register(DMLLMConfig, DMLLM)
