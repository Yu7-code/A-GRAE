---
dataset_info:
  features:
  - name: problem
    dtype: string
  - name: answer
    dtype: string
  splits:
  - name: test
    num_bytes: 11219
    num_examples: 30
  download_size: 10236
  dataset_size: 11219
configs:
- config_name: default
  data_files:
  - split: test
    path: data/test-*
---
