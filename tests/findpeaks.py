import numpy as np
from scipy.signal import find_peaks

# データを定義します（ここではサイン波を例とします）
x = np.linspace(0, 100, 1000)
y = np.sin(x)

# ピークを検出します
peaks, _ = find_peaks(y)

# ピークの高さを取得します
peak_heights = y[peaks]

# ピークの高さでソートし、最大と次に大きいピークを取得します
sorted_peaks = peaks[np.argsort(peak_heights)][-2:]
min_peaks = peaks[np.argsort(peak_heights)][:2]
test = sorted_peaks[::-1]

print(f"最大のピーク: x={x[sorted_peaks[1]]}, y={y[sorted_peaks[1]]}")
print(f"次に大きいピーク: x={x[sorted_peaks[0]]}, y={y[sorted_peaks[0]]}")
print(f"最小のピーク: x={x[min_peaks[0]]}, y={y[min_peaks[0]]}")
print(f"次に小さいピーク: x={x[min_peaks[1]]}, y={y[min_peaks[1]]}")