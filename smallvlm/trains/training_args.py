from typing import Optional
import os
from dataclasses import dataclass, fields, field
from transformers import TrainingArguments


@dataclass
class MLLMTrainingArguments(TrainingArguments):
    use_online_length_grouped_dataloader: bool = field(
        default=False,
        metadata={"help": "Whether to use grouping strategy to make the data in one batch have similar lengths."},
    )
    vision_model_lr: Optional[float] = field(
        default=None,
        metadata={"help": "Learning rate for vision model. If None, uses the global learning_rate."},
    )
    vision_abstractor_lr: Optional[float] = field(
        default=None,
        metadata={"help": "Learning rate for vision abstractor/projector. If None, uses the global learning_rate."},
    )
    language_model_lr: Optional[float] = field(
        default=None,
        metadata={"help": "Learning rate for language model. If None, uses the global learning_rate."},
    )
    max_input_length: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum input length for the model."},
    )
    def __init__(self, *args, **kwargs):
        unassigned_kwargs = kwargs.keys() - {field.name for field in fields(TrainingArguments)}
        unassigned_kwargs = {k: kwargs.pop(k) for k in unassigned_kwargs}

        super().__init__(*args, **kwargs)

        for key, value in unassigned_kwargs.items():
            setattr(self, key, value)

    def __post_init__(self):
        if self.logging_dir is None and self.output_dir is not None:
            self.logging_dir = os.path.join(self.output_dir, "runs", "logging")

        super().__post_init__()

    @property
    def is_main_process(self) -> bool:
        return self.process_index == 0
    