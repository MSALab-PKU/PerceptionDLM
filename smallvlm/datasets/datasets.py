from typing import Optional, Union, List
import traceback
import warnings
import time
import threading
import random
import bisect
import yaml
import torch
import copy
from transformers import ProcessorMixin

from smallvlm.utils.loggings import logger
from .database import load_database
# from .prompters import Prompter
# from .augmentations import Augmentation

def build_datasets(data_path: str,
                   processor: ProcessorMixin,
                   max_length: int,
                   augmentations: Optional[List[Union[str, dict]]] = None,
                   start_index: int = 0,  # --- 新增参数 ---
                   **kwargs) -> torch.utils.data.Dataset:
    """
    Args:
        ...
        start_index (int): The index to start training from (skip previous data).
        ...
    """
    if data_path.endswith('_datasets.yaml'):
        data_sources = _read_multi_datasets(data_path)
    else:
        data_sources = [{'data_file': data_path}]

    augmentation = None

    datasets = []
    num_samples = []
    for i, data_source in enumerate(data_sources):
        data_kwargs = kwargs.copy()
        data_kwargs.update(data_source)
        datafile = data_kwargs.pop('data_file')
        
        
        database = load_database(datafile, **data_kwargs)
        if len(database) == 0:
            data_kwargs.pop('num_samples', -1)
            continue

        num_samples.append(data_kwargs.pop('num_samples', -1))
        dataset = MMDataset(database, processor, augmentation, max_length)
        datasets.append(dataset)
    
    # --- 修改：将 start_index 传给 SampleDatasets ---
    dataset = SampleDatasets(datasets, num_samples, start_index=start_index)

    return dataset


def _read_multi_datasets(data_path):
    def _parse_dataset(data_source):
        if isinstance(data_source, dict):
            if 'data_file' in data_source:
                return [data_source]
            else:
                return sum((_parse_dataset(d) for d in data_source.values()), [])
        elif isinstance(data_source, list):
            return sum((_parse_dataset(d) for d in data_source), [])

    with open(data_path) as f:
        data_sources = yaml.load(f, Loader=yaml.FullLoader)

    data_sources = _parse_dataset(data_sources)

    return data_sources


class MMDataset(torch.utils.data.Dataset):
    """MultiModal dataset"""
    def __init__(self,
                 dataset,
                 processor,
                 augmentation,
                 max_length: int):
        self.dataset = dataset
        self.processor = processor
        self.augmentation = augmentation
        self.max_length = max_length
        self.max_warning_nums = 1
        self.warning_nums = 0
        ##debug
        self.failed_data = []
        self.failed_data_counter = {}
        ##debug

    def __getitem__(self, index):
        while True:
            record = None
            try:
                record = copy.deepcopy(self.dataset[index])
                prompt, generation_indices = self.processor.apply_chat_template(record['conversation'],
                                                            tokenize=True,
                                                            add_generation_prompt=False,
                                                            return_assistant_tokens_mask=True,
                                                            return_dict=True)
                with warnings.catch_warnings(record=True) as ws:
                    inputs = self.processor(text=prompt,
                                            generation_indices=generation_indices,
                                            images=record.get('images', []),
                                            truncation=True,
                                            max_length=self.max_length)
                    
                    seq_len = len(inputs['input_ids'][0])
                    assert len(inputs['input_ids'][0]) <= self.max_length, "inputs is too long"
                    inputs['labels'] = self.get_labels(inputs, seq_len)

                    num_img_tokens = (inputs['input_ids'][0] == self.processor.image_token_id).sum().item()
                    
                    is_all_zero = (torch.sum(inputs['pixel_values'] ) == 0).item()
                    assert not (is_all_zero and num_img_tokens > 0), "image is all zero while num_img_tokens > 0"

                    if len(ws) > 0 and self.warning_nums < self.max_warning_nums:
                        ws = '||'.join(str(w.message) for w in ws)
                        logger.warning(f"\nData Warning: {ws} #{record}")
                        self.warning_nums += 1

            except Exception as e:
                # if not isinstance(e, (FileNotFoundError, AssertionError, TimeoutError)) or self.inferring:
                if not isinstance(e, (FileNotFoundError, AssertionError, TimeoutError)):
                    logger.error(f"\n{traceback.format_exc()}")
                logger.warning(f"\n<DATA ERROR> <{type(e).__name__}: {e}> @{self.dataset.path}[{index}] #{record} ")
                time.sleep(0.1)
                index = random.randint(0, len(self))
            else:
                break
        return inputs


    def get_labels(self, inputs, seq_len):
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        prompt_mask = inputs.pop('prompt_mask')
        vision_mask = input_ids == self.processor.image_token_id

        labels = input_ids.clone()
        non_loss_mask = ~attention_mask | prompt_mask | vision_mask
        labels[non_loss_mask] = -100

        assert not (labels[:, 1:] == -100).all().item(), \
            "All labels are masked: " \
            f"{input_ids.shape} {non_loss_mask.shape} {seq_len} " \
            f"{torch.count_nonzero(~attention_mask)} " \
            f"{torch.count_nonzero(prompt_mask)} " \
            f"{torch.count_nonzero(vision_mask)}"
        return labels

    def prepare_doc_inputs(self, inputs):
        if len(inputs['pixel_values']) > 0:
            vision_mask = inputs['input_ids'] == self.processor.image_token_id
            inputs['attention_mask'] = \
                self.create_interleaved_attention_mask(inputs.pop('attention_mask'), vision_mask)

        return inputs

    @staticmethod
    def create_interleaved_attention_mask(attention_mask, vision_mask):
        vision_mask = vision_mask[0]
        vision_mask_ = ~(vision_mask & vision_mask.unsqueeze(1))

        attention_mask_ = torch.ones((1, 1, attention_mask.size(1), attention_mask.size(1)), dtype=torch.bool)
        attention_mask_.tril_()
        attention_mask_[0, 0, vision_mask.unsqueeze(0) & vision_mask_] = True
        attention_mask_[0, 0, vision_mask.unsqueeze(1) & vision_mask_] = False
        attention_mask_[0, attention_mask == False] = False

        return attention_mask_

    def __len__(self):
        return len(self.dataset)


class SampleDatasets(torch.utils.data.Dataset):
    def __init__(self, datasets: List[MMDataset], num_samples: List[int], start_index: int = 0):
        super().__init__()
        self.datasets = datasets
        assert len(self.datasets) > 0, 'datasets should not be an empty iterable'
        for d in self.datasets:
            assert not isinstance(d, torch.utils.data.IterableDataset), "ConcatDataset does not support IterableDataset"
        self.num_samples = num_samples
        self.cumulative_sizes = []
        self._samples = {}
        self._nth_epoch = -1
        self.start_index = start_index 
        self.set_epoch(0)
        total_len = self.cumulative_sizes[-1]

        if self.start_index >= total_len:
            logger.warning(f"Start index ({self.start_index}) is larger than dataset length ({total_len}). Dataset will be empty.")
        
        self._max_refetch = 5

    def __len__(self):
        return max(0, self.cumulative_sizes[-1] - self.start_index)

    def _rand_another(self):
        idx = random.randint(0, len(self) - 1)
        return idx

    def _get_data(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
            
        real_idx = idx + self.start_index
        
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, real_idx)

        sample_idx = real_idx if dataset_idx == 0 else real_idx - self.cumulative_sizes[dataset_idx - 1]
        if dataset_idx in self._samples:
            sample_idx = self._samples[dataset_idx][sample_idx]

        return self.datasets[dataset_idx][sample_idx]

    def __getitem__(self, index):
        for _ in range(self._max_refetch + 1):
            try:
                data = self._get_data(index)
            except Exception as e:
                logger.warning(f"Data error at index {index}: {e}")
                data = None

            if data is None:
                index_old = index
                index = self._rand_another()
                logger.warning(f"[WARNING] data {index_old} is None, use {index}!")
                continue
            return data
        raise RuntimeError(f"Failed to fetch data after {self._max_refetch} retries.")

    def set_epoch(self, nth_epoch):
        if nth_epoch != self._nth_epoch:
            self._sample_indices(nth_epoch)
        self._nth_epoch = nth_epoch

    def _sample_indices(self, nth_epoch):
        logger.info(f'resample datasets in epoch_{nth_epoch}')
        g = torch.Generator()
        g.manual_seed(281 + nth_epoch)
        cumulative_sizes = [0]
        for i, (dataset, num_samples) in enumerate(zip(self.datasets, self.num_samples)):
            if len(dataset) == 0:
                continue
            if num_samples >= 0:
                indices = torch.randperm(len(dataset), generator=g)[:num_samples % len(dataset) if num_samples % len(dataset) != 0 else len(dataset)].tolist()
                if num_samples > len(dataset):
                    logger.warning(f"num_samples ({num_samples}) exceed dataset size ({len(dataset)}), "
                                   f"duplicating the dataset.")
                    indices = list(range(len(dataset))) * (num_samples // len(dataset)) + indices
                self._samples[i] = indices
                cumulative_sizes.append(cumulative_sizes[-1] + num_samples)
            else:
                cumulative_sizes.append(cumulative_sizes[-1] + len(dataset))
        self.cumulative_sizes = cumulative_sizes[1:]