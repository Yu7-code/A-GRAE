import pandas as pd

# 定义Parquet文件路径和输出JSON文件路径
parquet_path = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangquan/rl/zq/RLVR-Decomposed/data/amc23/test.parquet"
json_path = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangquan/rl/zq/RLVR-Decomposed/data/amc23/test.json"

# 读取Parquet文件
df = pd.read_parquet(parquet_path, engine='pyarrow')

# 转换为JSON文件（orient='records'表示按行存储为JSON对象列表）
df.to_json(json_path, orient='records', lines=False, force_ascii=False)

print(f"转换完成！JSON文件已保存至：{json_path}")