---
dataset_info:
  features:
  - name: problem
    dtype: string
  - name: level
    dtype: string
  - name: type
    dtype: string
  - name: solution
    dtype: string
  splits:
  - name: train
    num_bytes: 5984772
    num_examples: 7500
  - name: test
    num_bytes: 3732833
    num_examples: 5000
  download_size: 4829140
  dataset_size: 9717605
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: test
    path: data/test-*
---
