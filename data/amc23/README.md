---
dataset_info:
  features:
  - name: id
    dtype: int64
  - name: problem
    dtype: string
  - name: answer
    dtype: string
  - name: url
    dtype: string
  splits:
  - name: test
    num_bytes: 14781
    num_examples: 40
  download_size: 11837
  dataset_size: 14781
configs:
- config_name: default
  data_files:
  - split: test
    path: data/test-*
---
