import os
import json
import glob
import torch
import shutil

from smallvlm.utils.parameter_manage import Parameters
from smallvlm.models import build_processor, build_model
from smallvlm.utils.model_utils import load_state_dict_file

def load_config(name_or_path: str):
    name_or_path = os.path.join(name_or_path, 'training_config.yaml')
    config = Parameters()
    config.merge_from_yaml(name_or_path)
    return config

def load_processor(config=None):
    processor = build_processor(config.PROCESSOR_CONFIG)
    return processor

def load_model(name_or_path: str,
               config = None,
               processor = None,
               attn_implementation = None):
    if processor is None:
        processor = load_processor(name_or_path, config=config)

    if attn_implementation is not None:
        config.MODEL_CONFIG['language_model']['attn_implementation'] = attn_implementation

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(config.MODEL_CONFIG, processor=processor, device=device)

    pretrained_model_path = glob.glob(os.path.join(name_or_path, '*.safetensors'))
    if pretrained_model_path is not None:
        load_state_dict_file(pretrained_model_path, model=model)
    
    return model


def copy_code(export_path):
    def get_name_or_path(config):
        paths = set()
        for k in list(config.keys()):
            if k == '_name_or_path':
                paths.add(config[k])
            elif isinstance(config[k], dict):
                paths.update(get_name_or_path(config[k]))
        return paths

    config_path = os.path.join(export_path, 'config.json')
    with open(config_path) as f:
        config = json.load(f)

    dependencies = list(get_name_or_path(config))
    for dep in dependencies:
        for file in os.listdir(dep):
            if file.endswith('.py') and file != '__init__.py':
                if os.path.exists(os.path.join(export_path, file)):
                    continue
                shutil.copy(os.path.join(dep, file), export_path)


def mod_config(export_path):
    def remove_name_or_path(config):
        for k in list(config.keys()):
            if k == '_name_or_path':
                config.pop(k)
            elif isinstance(config[k], dict):
                remove_name_or_path(config[k])

    config_path = os.path.join(export_path, 'config.json')
    with open(config_path) as f:
        config = json.load(f)

    remove_name_or_path(config)

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)


def main(model_path, export_path):
    config = load_config(model_path)
    processor = load_processor(config)
    if not getattr(processor, '_auto_class', None):
        processor._auto_class = "AutoProcessor"

    processor.save_pretrained(export_path)

    model = load_model(model_path, config, processor=processor)
    if not getattr(model.config, '_auto_class', None):
        model.config._auto_class = "AutoConfig"
    if not getattr(model, '_auto_class', None):
        model._auto_class = "AutoModelForCausalLM"

    model.save_pretrained(export_path, max_shard_size='4GB', safe_serialization=True)

    # copy_code(export_path)

    # mod_config(export_path)


if __name__ == "__main__":
    from argparse import ArgumentParser

    def parse_args():
        parser = ArgumentParser()
        parser.add_argument('--model_path', type=str, default='work_dirs/s2')
        parser.add_argument('--export_path', type=str, default='work_dirs/s2/exported_model')

        return parser.parse_args()

    args = parse_args()

    main(args.model_path, args.export_path)