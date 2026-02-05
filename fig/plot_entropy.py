# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # 1. 读取数据
# df = pd.read_csv("entropy_math.csv")

# # 2. 全局绘图配置（提升美观性）
# plt.rcParams["font.family"] = "Arial"  # 统一字体
# plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
# sns.set_style("whitegrid", {"grid.linestyle": "--", "grid.alpha": 0.6})  # 浅灰色虚线网格

# # 3. 创建画布（比例更协调）
# fig, ax = plt.subplots(figsize=(12, 7), dpi=100)

# # 4. 定义高区分度配色+差异化线型（避免单靠颜色区分，适配色盲友好）
# colors = {
#     "Positive-Dominant": "#E63946",  # 暖红（高饱和）
#     "Negative-Dominant": "#1D3557",  # 深海蓝（低亮度）
#     "GRPO": "#457B9D"         # 浅湖蓝（中等亮度）
# }
# linestyles = {
#     "Positive-Dominant": "-",   # 实线（突出主要指标）
#     "Negative-Dominant": "--",  # 虚线（次要指标）
#     "GRPO": "-."         # 点划线（参考指标）
# }
# linewidths = {
#     "Positive-Dominant": 2.5,   # 略粗突出
#     "Negative-Dominant": 2.0,
#     "GRPO": 2.0
# }

# # 5. 绘制折线图（无端点标记，仅连线）
# for col in ["Positive-Dominant", "Negative-Dominant", "GRPO"]:
#     sns.lineplot(
#         data=df,
#         x="Step",
#         y=col,
#         color=colors[col],
#         linestyle=linestyles[col],
#         linewidth=linewidths[col],
#         ax=ax,
#         label=col,
#         marker="",  # 强制无标记点（确保无端点）
#         alpha=0.85   # 轻微透明，避免线条重叠时过亮
#     )

# # 6. 精细化美化
# # 标题/轴标签（字号+间距）
# ax.set_title("Trend of Entropy Metrics", fontsize=16, pad=20, fontweight="bold")
# ax.set_xlabel("Step", fontsize=14, labelpad=12)
# ax.set_ylabel("Value", fontsize=14, labelpad=12)

# # 轴刻度（字号+样式）
# ax.tick_params(axis="both", labelsize=12, length=6, width=1.2)

# # 图例（位置+样式）
# ax.legend(
#     loc="best",
#     fontsize=12,
#     frameon=True,
#     fancybox=True,
#     shadow=True,
#     framealpha=0.9,
#     borderpad=1
# )

# # 移除多余边框+调整网格
# ax.spines["top"].set_visible(False)
# ax.spines["right"].set_visible(False)
# ax.spines["left"].set_linewidth(1.2)
# ax.spines["bottom"].set_linewidth(1.2)
# ax.grid(axis="y", alpha=0.6)  # 仅保留y轴网格，更简洁

# # 7. 保存/显示（高清+无白边）
# plt.tight_layout()
# plt.savefig("entropy_trend_optimized.png", dpi=300, bbox_inches="tight")
# plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ===================== 1. 全局样式配置（与参考图完全一致） =====================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']  # 中文字体+英文
plt.rcParams['axes.unicode_minus'] = False            # 解决负号显示
plt.rcParams['axes.linewidth'] = 1.2                  # 边框线宽
plt.rcParams['grid.alpha'] = 0.3                      # 网格透明度
plt.rcParams['grid.linestyle'] = '--'                 # 网格线型

# ===================== 2. 数据读取与预处理 =====================
# 读取数据（确保路径正确）
df = pd.read_csv("entropy_math.csv")
# 打印列名确认（方便排查列名问题）
print("CSV列名：", df.columns.tolist())

# 定义绘图列名（根据实际CSV列名调整！）
# 示例：如果列名是Step, Positive-Dominant, Negative-Dominant, GRPO，则：
x_col = "Step"
y_cols = ["Positive-Dominant", "Negative-Dominant", "GRPO"]
# 方法名（图例显示）
method_names = ['Positive-Dominant', 'Negative-Dominant', 'GRPO']

# ===================== 3. 样式参数（与参考图一致） =====================
# 配色（沿用参考图的高区分度配色）
colors = ['#FF5252', '#2196F3', '#4CAF50']
# 线型（参考图风格，无标记点以实现“无端点”）
linestyles = ['-', '--', '-.']
linewidths = [2.5, 2.5, 2.5]  # 与参考图linewidth=2.5一致
alpha = 0.9                    # 透明度与参考图一致

# ===================== 4. 绘图 =====================
# 创建画布（尺寸与参考图一致：12,8）
plt.figure(figsize=(12, 8))
ax = plt.gca()

# 绘制折线（无标记点=无端点，完全匹配参考图风格）
for i, (y_col, name, color, ls, lw) in enumerate(zip(y_cols, method_names, colors, linestyles, linewidths)):
    ax.plot(
        df[x_col], df[y_col],
        label=name,
        color=color,
        linestyle=ls,
        linewidth=lw,
        alpha=alpha,
        marker="",  # 强制无标记点（核心：不显示端点）
        markersize=0  # 双重保障：标记大小为0
    )

# ===================== 5. 样式美化（与参考图完全对齐） =====================
# 坐标轴标签（字体大小+加粗）
ax.set_xlabel('Step', fontsize=14, fontweight='bold')
ax.set_ylabel('Entropy Value', fontsize=14, fontweight='bold')

# 标题（参考图风格）
ax.set_title('Entropy on MATH Dataset (Test Set)', fontsize=16, fontweight='bold', pad=20)

# 网格（参考图风格：alpha=0.3，虚线）
ax.grid(True, alpha=0.3, linestyle='--')

# 背景色（参考图的浅灰色）
ax.set_facecolor('#f8f9fa')

# 边框美化（参考图风格：线宽1.2）
for spine in ax.spines.values():
    spine.set_linewidth(1.2)

# 图例（参考图位置+样式）
ax.legend(loc='upper left', fontsize=12, framealpha=0.9)

# 坐标轴刻度字体大小（参考图12号字）
ax.tick_params(axis='both', labelsize=12)

# ===================== 6. 保存与显示 =====================
plt.tight_layout()  # 紧凑布局
# 保存（高清300dpi，无白边，参考图命名风格）
plt.savefig('Entropy_math.png', dpi=300, bbox_inches='tight')
plt.show()