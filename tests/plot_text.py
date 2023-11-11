import matplotlib.pyplot as plt
import numpy as np

# データを生成します
data = np.random.randn(100)

# 平均値を計算します
mean = np.mean(data)

# フォントサイズ指定
fontsize = 10

# ヒストグラムをプロットします
plt.hist(data, bins=20)

# グラフの下部にスペースを追加
plt.subplots_adjust(bottom=fontsize/72)
# 2行設ける場合...
#bs = fontsize/72
#plt.subplots_adjust(bottom=bs+bs/4)

# 平均値をテキストで表示します
plt.figtext(0.98, 0.02+(fontsize/72/4), 'average: {:.2f}'.format(mean), ha='right', fontsize=fontsize)
plt.figtext(0.98, 0.02, 'average: {:.2f}'.format(mean), ha='right', fontsize=fontsize)

# グラフを表示します
plt.show()
