
from transformers import TrainingArguments, TrainerState, TrainerControl
from transformers.trainer_callback import ProgressCallback, TrainerCallback
from .utils import Timer


class TimingProgressCallback(ProgressCallback):
    """
    A [`TrainerCallback`] that displays the progress of training or evaluation.
    """
    def __init__(self, timer: Timer = None):
        super().__init__()
        self.timer = Timer() if timer is None else timer

    def on_step_begin(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self.timer.stop('DataTime')

    def on_step_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self.timer.stop('TotalTime')
            self.training_bar.update(self.get_step(state) - self.current_step)
            self.training_bar.set_postfix_str(str(self.timer), False)
            self.current_step = self.get_step(state)
            self.timer.start('DataTime')
            self.timer.start('TotalTime')

    def on_substep_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self.timer.start('DataTime')
            self.timer.start('TotalTime')

    @staticmethod
    def get_step(state):
        return state.global_step


