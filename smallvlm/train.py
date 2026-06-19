import sys
import os
import pathlib
from smallvlm.trains import MLLMTrainer, MLLMTrainingArguments
from smallvlm.utils.parameter_manage import ArgumentParser, Parameters
from smallvlm.datasets.collator import Collator
from smallvlm.datasets.datasets import build_datasets
from smallvlm.models import build_processor, build_model
from smallvlm.utils.loggings import logger


def build(config, data_path):
    training_args = MLLMTrainingArguments(**config.TRAINING_CONFIG, max_input_length=config.DATA_CONFIG['max_length'])

    if training_args.is_main_process:
        logger.info(f"<<TRAINING CONFIG>>\n{config}")
        if training_args.output_dir:
            if not os.path.exists(training_args.output_dir):
                os.makedirs(training_args.output_dir)
            config.dump(os.path.join(training_args.output_dir, 'training_config.yaml'))

    logger.info(">>> Building Data...")
    processor = build_processor(config.PROCESSOR_CONFIG)
    if 'pdmllm' in config.MODEL_CONFIG.get('architecture', None):
        from smallvlm.datasets.multi_mask_datasets import build_mask_datasets
        train_data = build_mask_datasets(data_path, processor=processor, **config.DATA_CONFIG)
    else:
        train_data = build_datasets(data_path, processor=processor, **config.DATA_CONFIG)
    
    if training_args.is_main_process:
        from shutil import copyfile
        copyfile(data_path, os.path.join(training_args.output_dir, os.path.basename(data_path)))

    collator = Collator(pad_token_id=processor.pad_token_id)

    # from torch.utils.data import DataLoader
    # dataloader = DataLoader(train_data, batch_size=2, collate_fn=collator, shuffle=True, num_workers=0)
    # from smallvlm.datasets.dataloaders_olg import apply_online_length_grouped_dataloader
    # if 'DataLoaderAdapter' in iter(c.__name__ for c in type(dataloader).__mro__):
    #     dataloader_ = dataloader.base_dataloader
    # else:
    #     dataloader_ = dataloader
    # apply_online_length_grouped_dataloader(dataloader_)
    # for i, batch in enumerate(dataloader):
    #     print(batch)

    logger.info(">>> Building Model...")
    model = build_model(config.MODEL_CONFIG, processor=processor, device=training_args.device)
    
    logger.info(">>> Building Trainer...")
    trainer = MLLMTrainer(model=model,
                          train_dataset=train_data, data_collator=collator,
                          args=training_args)

    return trainer


def main(config, data_path):
    logger.info(">>> Begin Building...")
    trainer = build(config, data_path=data_path)

    logger.info(">>> Begin Training...")
    
    if list(pathlib.Path(trainer.args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_model()

if __name__ == "__main__":
    def parse_args():
        parser = ArgumentParser()
        parser.add_argument('--data_path', required=True, type=str)
        parser.add_argument('--config', default=None, type=str)

        return parser.parse_known_args()

    def parse_more_args(more_args):
        more_config = {}
        for key, value in more_args.items():
            ks = key.split('.')
            if ks[0] not in {'TRAINING_CONFIG', 'DATA_CONFIG', 'PROCESSOR_CONFIG', 'MODEL_CONFIG'}:
                continue
            dict_ = more_config
            for k in ks[:-1]:
                if k not in dict_:
                    dict_[k] = {}
                dict_ = dict_[k]
            dict_[ks[-1]] = value

        return more_config

    args, more_args = parse_args()
    more_config = parse_more_args(more_args)  # Load config parameters from input arguments.

    config = Parameters()
    if args.config:
        config.merge_from_yaml(args.config)
    config.merge_from_dict(more_config)
    
    main(config, data_path=args.data_path)
