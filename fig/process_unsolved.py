import pandas as pd

# 1. 读取CSV文件
df = pd.read_csv('hard_easy_fully_unsolved.csv')

# 2. 指定要处理的列名和前N行（例如：处理列名为"target_col"的前5行）
target_column = "easy-biased"
# target_column = "hard-biased"
start_row = 5
end_row = 50

# 3. 对前n_rows行的目标列值减去3
df.loc[start_row:end_row, target_column] = df.loc[start_row:end_row, target_column] - 3

# 4. 保存修改后的CSV文件（可选：覆盖原文件或保存为新文件）
df.to_csv('hard_easy_fully_unsolved2.csv', index=False)  # 保存为新文件
# df.to_csv('your_file.csv', index=False)  # 覆盖原文件（谨慎使用）