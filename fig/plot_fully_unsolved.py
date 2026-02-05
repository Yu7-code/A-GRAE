import pandas as pd
import matplotlib.pyplot as plt

# 1. 读取 CSV 文件
df = pd.read_csv('hard_easy_math.csv')

ax = plt.gca() 
ax.set_xlim(left=7)
# ax.spines['left'].set_position('zero')
# ax.spines['bottom'].set_position('zero')

# 2. 设置图片清晰度和尺寸
plt.rcParams['figure.dpi'] = 300
plt.figure(figsize=(10, 6))

# 3. 绘制各列的折线图（指定不同颜色，GRPO用红色）
plt.plot(df['Step'], df['easy-biased'], label='easy-biased', color='blue',  linewidth=1.5)
plt.plot(df['Step'], df['hard-biased'], label='hard-biased', color='green',  linewidth=1.5)
plt.plot(df['Step'], df['grpo'], label='GRPO', color='red',  linewidth=1.5)

# 4. 添加图表标签和图例
plt.xlabel('Step')
# plt.ylabel('Unsolved Problems')
# plt.title('Fully Unsolved Problems in Training Process')
plt.ylabel('Test Accuracy')
plt.title('Test Accuracy in Training Process')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# 5. 保存并显示图表
plt.savefig('hard_easy_accuracy.png', bbox_inches='tight')
plt.show()