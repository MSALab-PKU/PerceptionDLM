# Training Datasets

This document details the data formats and directory locations for training **PerceptionDLM-Base** and **PerceptionDLM**.
Data configurations can be found inside `configs/dmllm/data_configs` and `configs/pdmllm/data_configs`.

## Training Stages (PerceptionDLM-Base)

Training for PerceptionDLM-Base uses data inside the `datasets/` directory across 4 distinct stages. Configurations are located under `configs/dmllm/data_configs/(s1,s2,s3,s4)/`.

### Stage Examples
An example YAML (e.g., `s1/honey_stage1_1M_datasets.yaml`) looks like this:

```yaml
data:
- data_file: datasets/Bee-Training-Data-Stage1/data
  data_type: parquet
  num_samples: 994839
  row_size: 16
```
Other stages reference their respective directories: `Bee-Training-Data-Stage2`, `LLaVA-OneVision-1.5-Instruct-Data`, and `Honey-Data-15M`.

## Training PerceptionDLM

The `PerceptionDLM` model utilizes region-mask annotation data found in the `annotations/` directory alongside `images/`. The configurations are located in `configs/pdmllm/data_configs/`.

### Config Example (`PerceptionDLM_datasets.yaml`):
```yaml
gar_caption:
- data_file: annotations/dam_dataset.json
  data_type: json
- data_file: annotations/coconut_dataset.json
  data_type: json
- data_file: annotations/sam_dataset.json
  data_type: json
```
