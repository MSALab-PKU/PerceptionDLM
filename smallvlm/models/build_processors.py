import importlib
from transformers import AutoTokenizer, AutoImageProcessor, ProcessorMixin

from smallvlm.utils.parameter_manage import Parameters


def build_processor(config: dict) -> ProcessorMixin:
    config = config.copy()

    tokenizer_config = config.pop('tokenizer_config')
    tokenizer_path = tokenizer_config.pop('name_or_path')
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, **tokenizer_config, trust_remote_code=True)
    
    image_processor_config = config['image_processor_config']
    architecture = image_processor_config.pop('architecture', None)
    if architecture is not None:
        ImageProcessor = getattr(importlib.import_module(f"smallvlm.models.{'.'.join(architecture.split('.')[:-1])}"), architecture.split('.')[-1])
        image_processor = ImageProcessor(**image_processor_config)
    else:
        image_processor_path = image_processor_config.pop('name_or_path')
        image_processor = AutoImageProcessor.from_pretrained(
            image_processor_path, **image_processor_config, trust_remote_code=True)


    processor_class = config.pop('processor_class', None)
    if processor_class is None:
        raise ValueError("Processor class (processor_class) is not given.")
    Processor = getattr(importlib.import_module(f"smallvlm.models.{'.'.join(processor_class.split('.')[:-1])}"), processor_class.split('.')[-1])

    processor = Processor(tokenizer=tokenizer, image_processor=image_processor, **config)

    return processor

