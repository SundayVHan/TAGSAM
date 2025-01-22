import matplotlib.pyplot as plt
import numpy as np

# 数据
x = [2, 4, 8, 16]
y1 = [30, 45, 40, 35]
y2 = [35, 40, 38, 30]
y3 = [25, 35, 28, 23]

# 创建图形
plt.figure(figsize=(5, 6))

# 绘制阴影区域
plt.axvspan(3, 5, color='red', alpha=0.2)  # 从x=3到x=5填充红色阴影

# 绘制折线图
plt.plot(x, y1, color='orange', marker='o', label='Line 1')
plt.plot(x, y2, color='red', marker='o', label='Line 2')
plt.plot(x, y3, color='blue', marker='o', label='Line 3')

# 设置标签和标题
plt.xlabel('Epochs')
plt.ylabel('Top-1 Accuracy (%)')

# 设置坐标轴刻度
plt.xticks(x)
plt.yticks(np.arange(20, 50, 5))

# 添加网格
plt.grid(alpha=0.3)

# 添加图例
plt.legend()

# 显示图形
plt.tight_layout()
plt.savefig("./fig.png")
