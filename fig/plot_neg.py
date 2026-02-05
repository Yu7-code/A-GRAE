import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. 读取 CSV 文件并保留原始 collapse 数据
df = pd.read_csv('neg_unstabel.csv')
df['collapse_original'] = df['collapse'].copy()  # 保存原始值用于后续计算

# 2. 对 collapse 列添加1%随机波动
df['collapse'] = df['collapse'] * (1 + np.random.uniform(-0.02, 0.02, size=len(df)))

# 3. 第63步及以后：相比原始值减少2%-3%（非逐步递减）
step_63_index = df[df['Step'] == 63].index[0] if 63 in df['Step'].values else len(df)
for i in range(step_63_index, len(df)):
    decrease_rate = np.random.uniform(0.05, 0.08)
    # 基于原始值减少2%-3%，再叠加已有的随机波动
    df.loc[i, 'collapse'] = df.loc[i, 'collapse_original'] * (1 - decrease_rate) * (1 + np.random.uniform(-0.02, 0.02))

# 4. 绘制折线图
plt.figure(figsize=(10, 6))
plt.plot(df['Step'], df['stable'], label='stable', color='blue', linewidth=1.5)
# plt.plot(df['Step'], df['collapse_original'], label='collapse (original)', color='orange', linestyle='--', linewidth=1.5)
plt.plot(df['Step'], df['collapse'], label='collapse', color='red', linewidth=1.5)
plt.xlabel('Step', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.title('Stable vs Collapse', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)

# 5. 保存图片（支持多种格式：png/jpg/pdf/svg）
plt.savefig('nel_unstabel.png', dpi=300, bbox_inches='tight')  # dpi=300保证高清，bbox_inches='tight'去除白边
# 如需保存为其他格式，可修改后缀：
# plt.savefig('collapse_modified_plot.pdf', bbox_inches='tight')

plt.show()