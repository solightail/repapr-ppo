""" Quote from repapr """
import numpy as np

class Formula:
    """ Formula used to calculate PAPR """
    def __init__(self, tones: int, del_freq: float, amp: float) -> None:
        # theta_k の初期化はしないようにすることで、インスタンス生成を1回で済むように変更
        self.tones: int = tones
        self.del_freq: float = del_freq
        self.amp: float = amp

    def calc_p0t(self, time, theta_k_values):
        """ P_0(t) 計算 """
        # 計算プログラム自体は ChatGPT に丸投げし、修正を加えたもの

        # 変数定義 (numpyにて公差を求め、2次元配列として定義)
        # np.arange で指定した上限値は未満となることに注意！
        k_values = np.arange(1, self.tones)    # k=1 -> k=N-1
        l_values = np.arange(2, self.tones+1)  # l=2 -> l=N
        l_values_h = l_values[:, np.newaxis]   # l_valuesを2次元に変形

        # cos 計算 (2重和に向けて2次元配列のまま計算を行っている)
        # np.take -> 与えられたインデックスに従って配列から要素を選択する関数
        # l_values_h - 1 と k_values - 1 は配列から選択するための数値調整
        cos_values = np.cos(2 * np.pi * (l_values_h - k_values) * self.del_freq * time
                            + np.take(theta_k_values, l_values_h - 1)
                            - np.take(theta_k_values, k_values - 1))

        # 2重和計算
        p0t = np.sum(cos_values * (l_values_h > k_values))
        return p0t

    def calc_ept(self, p0t):
        """ 瞬時包絡線電力 計算 """
        ept = self.tones * self.amp**2 + 2 * self.amp**2 * p0t
        return ept

    def calc_papr_w(self, p0t):
        """ PAPR（Peak-to-Average-Power-Ratio）計算 """
        papr_w = 1 + ((2 * p0t) / self.tones)
        return papr_w

class FEPtA:
    """ Formula クラスを移譲して、各式より求められる値をリストにまとめる """
    def __init__(self, formula: Formula, del_time: float) -> None:
        self.del_time = del_time
        self.formula = formula
        self.time_points = np.arange(0.0, 1.0 + self.del_time, self.del_time)

    def get_ept(self, theta_k_values):
        """ ep_t_arr のみ算出 """
        # 時間については ndarray で計算できる代物でないため、内包表現を用いて計算を行う
        p0t_arr = np.array([self.formula.calc_p0t(i, theta_k_values) for i in self.time_points])
        ept_arr = self.formula.calc_ept(p0t_arr)
        return ept_arr
    
    def get_ept_papr(self, theta_k_values):
        """ PAPR[dB] の算出まで """
        # 時間については ndarray で計算できる代物でないため、内包表現を用いて計算を行う
        p0t_arr = np.array([self.formula.calc_p0t(i, theta_k_values) for i in self.time_points])
        ept_arr = self.formula.calc_ept(p0t_arr)

        # PAPR最大値の算出
        max_p0t: float = np.max(p0t_arr)
        max_ept: float = np.max(ept_arr)
        max_papr_w: float = self.formula.calc_papr_w(max_p0t)
        max_papr_db: float = 10 * np.log10(max_papr_w)

        return ept_arr, max_ept, max_papr_w, max_papr_db
