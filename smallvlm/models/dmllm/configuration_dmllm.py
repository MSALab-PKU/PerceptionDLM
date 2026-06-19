from transformers import PretrainedConfig, AutoConfig, CONFIG_MAPPING
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.utils import logging

logger = logging.get_logger(__name__)

class DMLLMConfig(PretrainedConfig):
    model_type = "dmllm"
    is_composition = True

    def __init__(self,
                 language_model_config=None,
                 vision_model_config=None,
                 vision_abstractor_config=None,
                 image_token_id=None,
                 image_size=512,
                 patch_size=16,
                 downsample_ratio=0.5,
                 vision_select_layer=-2,
                 replacement_noise_mode=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.replacement_noise_mode = replacement_noise_mode
        self.image_size = image_size
        self.patch_size = patch_size
        self.downsample_ratio = downsample_ratio
        self.num_image_token = int((image_size // patch_size) ** 2 * (downsample_ratio ** 2))
        self.vision_select_layer = vision_select_layer

        if isinstance(language_model_config, dict):
            if '_name_or_path' not in language_model_config:
                language_model_config['_name_or_path'] = self._name_or_path
            language_model_type = language_model_config.get('model_type', '')
            is_remote_code = '.' in language_model_config.get('auto_map', {}).get('AutoConfig', '')
            if language_model_type in CONFIG_MAPPING and not is_remote_code:
                language_model_config = AutoConfig.for_model(**language_model_config)
            elif language_model_type:
                Config = get_class_from_dynamic_module(language_model_config["auto_map"]["AutoConfig"],
                                                       language_model_config['_name_or_path'])
                language_model_config = Config(**language_model_config)
        self.language_model_config = language_model_config

        if isinstance(vision_model_config, dict):
            if '_name_or_path' not in vision_model_config:
                vision_model_config['_name_or_path'] = self._name_or_path
            vision_model_type = vision_model_config.get('model_type', '')
            is_remote_code = '.' in vision_model_config.get('auto_map', {}).get('AutoConfig', '')
            if vision_model_type in CONFIG_MAPPING and not is_remote_code:
                vision_model_config = AutoConfig.for_model(**vision_model_config)
            elif vision_model_type:
                Config = get_class_from_dynamic_module(vision_model_config["auto_map"]["AutoConfig"],
                                                       vision_model_config['_name_or_path'])
                vision_model_config = Config(**vision_model_config)
        self.vision_model_config = vision_model_config

        self.vision_abstractor_config = vision_abstractor_config

        self.image_token_id = image_token_id

    @property
    def hidden_size(self):
        return self.language_model_config.hidden_size

    def to_dict(self):
        ret_dict = super().to_dict()
        ret_dict["auto_map"] = {
            "AutoConfig": "configuration_dmllm.DMLLMConfig",
            "AutoModel": "modeling_dmllm.DMLLM",
            "AutoModelForCausalLM": "modeling_dmllm.DMLLM"
        }
        return ret_dict

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        if 'name_or_path' in kwargs:
            config_dict['_name_or_path'] = kwargs.pop('name_or_path')
        return super().from_dict(config_dict, **kwargs)
