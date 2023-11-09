import matplotlib.pyplot as plt
import numpy as np

# データを生成します
data = np.random.randn(100)

# 平均値を計算します
mean = np.mean(data)

# ヒストグラムをプロットします
plt.hist(data, bins=20)

# 平均値をテキストで表示します
plt.figtext(0.975, 0.025, 'average: {:.2f}'.format(mean), ha='right')

# グラフを表示します
plt.show()
