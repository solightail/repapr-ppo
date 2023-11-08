from . import algorithms as algo
from . import calc

import numpy as np
import gym.spaces
import torch.nn as nn
from scipy.signal import find_peaks

class MtEnv(gym.Env):
    def __init__(self, tones: int, del_freq:float, del_time:float, amp:float,
                 max_step: int, action_div: float, action_list: list, reward_x: float,
                 init_model: str='random', manual: list=None, re_model: str='USo_v0',):
        super().__init__()

        self.tones: int = tones
        self.del_freq: float = del_freq
        self.del_time: float = del_time
        self.amp: float = amp
        self.max_step: int = max_step
        self.action_div: float = action_div
        self.action_list: list = action_list
        self.reward_x: float = reward_x
        self.init_model: str = init_model
        self.manual: float = manual
        self.re_model: str = re_model

        self.theta_k_values: np.array[float]
        self.time_values: np.array = np.arange(0.0, 1.0 + self.del_time, self.del_time)

        match self.re_model:
            case 'USa_v0':
                self.input_dims = 1
            case 'USo_v0':
                self.input_dims = 1
            case 'USt_v0':
                self.input_dims = 2
            case 'UFtSt_v0':
                self.input_dims = 2
            case 'BSa_v0':
                self.input_dims = 2
            case 'BSo_v0':
                self.input_dims = 2
            case 'BSt_v0':
                self.input_dims = 4
            case 'BFt_v0':
                self.input_dims = 4
            case 'USa_v0t':
                self.input_dims = 2
            case 'USo_v0t':
                self.input_dims = 2
            case 'USt_v0t':
                self.input_dims = 2
            case 'UFtSt_v0t':
                self.input_dims = 2
            case 'BSa_v0t':
                self.input_dims = 4
            case 'BSo_v0t':
                self.input_dims = 4
            case 'BSt_v0t':
                self.input_dims = 4
            case 'BFt_v0t':
                self.input_dims = 4

            case 'USo_v1':
                self.input_dims = 1
            case 'USt_v1':
                self.input_dims = 2
            case 'UFtSt_v1':
                self.input_dims = 2
            case 'BSo_v1':
                self.input_dims = 2
            case 'BSt_v1':
                self.input_dims = 4
            case 'BFt_v1':
                self.input_dims = 4

            case 'USto_v1':
                self.input_dims = 2
            case 'UFtSto_v1':
                self.input_dims = 2
            case 'BSto_v1':
                self.input_dims = 2
            case 'BStt_v1':
                self.input_dims = 4
            case 'BFto_v1':
                self.input_dims = 2
            case 'BFtt_v1':
                self.input_dims = 4

            case 'dB_v0':
                self.input_dims = tones+1

            case 'MSEa_v0':
                self.input_dims = 10000

        self.input_dims = (self.input_dims, )

        # action_space, observation_space, reward_range を設定する
        self.action_arr = np.array([self.action_list] * self.tones)
        self.n_action = len(self.action_list) ** self.tones
        self.action_space = gym.spaces.Discrete(self.n_action)
        self.init_observation = np.ones(self.input_dims) * (self.tones**2)
        self.observation_space = gym.spaces.Box(low=0, high=self.tones**2, shape=self.init_observation.shape)
        self.reward_range = np.array([0, 1*self.reward_x])
        self.reset()


    def reset(self):
        self.steps = 0
        # アルゴリズム選択
        match self.init_model:
            case 'all0':
                strategy = algo.All0(self.tones)
            case 'narahashi':
                strategy = algo.Narahashi(self.tones)
            case 'newman':
                strategy = algo.Newman(self.tones)
            case 'random':
                strategy = algo.Random(self.tones)
            case 'manual':
                strategy = algo.Manual(self.tones, self.manual)

        # theta_k 計算
        algo_context = algo.AContext(strategy)
        self.theta_k_values = algo_context.calc_algo()

        # observation 計算
        self.ep_t_array, self.max_ep_t, self.max_papr_w, self.max_papr_db = self._eptarr()
        observation, _, _ = self._obreward()

        return observation, None


    def step(self, action):
        # --- exec action ---
        # action を各トーンごとに分解
        each_action_tuple = np.unravel_index(action, (len(self.action_list),) * self.tones)
        for i in range(self.tones):
            self.theta_k_values[i] = self.theta_k_values[i] + ((self.action_arr[i][each_action_tuple[i]])*2*np.pi*self.action_div)
        self.ep_t_array, self.max_ep_t, self.max_papr_w, self.max_papr_db = self._eptarr()

        # --- observation & reward ---
        observation, reward_raw, up1h = self._obreward()
        reward = reward_raw * self.reward_x
        '''
        if np.all(each_action_tuple == 1):
            # actions において、1は停止となる
            reward = 0
        '''

        # terminated
        # 時間制限による終了
        self.steps += 1
        if (self.steps >= self.max_step):
            terminated = True
        else:
            terminated = False

        # truncated
        # エリア外等による強制終了
        if (up1h <= self.tones+2 or self.tones**2 <= up1h):
            truncated = True
        else:
            truncated = False

        return observation, reward, terminated, truncated, None


    def render(self, mode='human', close=False):
        """ utils.py に必要な要素を入れているのでパス """
        pass


    def _eptarr(self):
        formula = calc.Formula(self.tones, self.del_freq, self.amp)
        fepta = calc.FEPtA(formula, self.del_time)
        return fepta.get_ept_papr(self.theta_k_values)

    def _obreward(self):
        # ピーク値取得
        upper_peaks, _ = find_peaks(self.ep_t_array, distance=10, plateau_size=1)
        lower_peaks, _ = find_peaks(-self.ep_t_array, distance=10, plateau_size=1)

        # find_peaks は t = 0 にピークがある場合を考慮していない。
        # 1周期でこれを判断することは不可能であるため、2周期分で判断をし、1周期分に戻す作業を行う。
        two_cycle_ep_t_array = np.concatenate([self.ep_t_array, self.ep_t_array[1:]])
        two_cycle_upper_peaks, _ = find_peaks(two_cycle_ep_t_array, distance=10, plateau_size=1)
        two_cycle_lower_peaks, _ = find_peaks(-two_cycle_ep_t_array, distance=10, plateau_size=1)
        period: int = len(self.ep_t_array)-1
        if (period in two_cycle_upper_peaks):
            upper_peaks = np.concatenate([[0], upper_peaks, [period]])
        if (period in two_cycle_lower_peaks):
            lower_peaks = np.concatenate([[0], lower_peaks, [period]])

        # find_peaks 後処理
        # 最大ピークおよび準最大ピークの検出
        upper_peaks_heights = np.take(self.ep_t_array, upper_peaks)
        max2_reverse_upper_peaks = upper_peaks[np.argsort(upper_peaks_heights)][-2:]
        lower_peaks_heights = np.take(self.ep_t_array, lower_peaks)
        max2_reverse_lower_peaks = lower_peaks[np.argsort(lower_peaks_heights)][:2]

        # upper_peaks 平均・最大ピーク・準最大ピーク
        upah = np.average(np.take(self.ep_t_array, upper_peaks))
        up1t = np.take(self.time_values, max2_reverse_upper_peaks[1])
        up1h = np.take(self.ep_t_array, max2_reverse_upper_peaks[1])
        up2t = np.take(self.time_values, max2_reverse_upper_peaks[0])
        up2h = np.take(self.ep_t_array, max2_reverse_upper_peaks[0])

        # lower_peaks 平均・最大ピーク・準最大ピーク
        loah = np.average(np.take(self.ep_t_array, lower_peaks))
        lo1t = np.take(self.time_values, max2_reverse_lower_peaks[1])
        lo1h = np.take(self.ep_t_array, max2_reverse_lower_peaks[1])
        lo2t = np.take(self.time_values, max2_reverse_lower_peaks[0])
        lo2h = np.take(self.ep_t_array, max2_reverse_lower_peaks[0])

        # モデル別 observation & reward の算出
        match self.re_model:
            case 'USa_v0':
                observation = np.array([upah])
                reward = (self.tones**2 - upah) / (self.tones**2)
            case 'USo_v0':
                observation = np.array([up1h])
                reward = (self.tones**2 - up1h) / (self.tones**2)
            case 'USt_v0':
                observation = np.array([up1h, up2h])
                reward = ((self.tones**2 - up1h) + (self.tones**2 - up2h)) / (2 * self.tones**2)
            case 'UFtSt_v0':
                observation = np.array([up1h, up2h])
                reward = ((self.tones**2 - up2h) - (up1h - up2h)) / (self.tones**2)
            case 'BSa_v0':
                observation = np.array([upah, loah])
                reward = ((self.tones**2 - upah) + loah) / (2 * self.tones**2)
            case 'BSo_v0':
                observation = np.array([up1h, lo1h])
                reward = ((self.tones**2 - up1h) + lo1h) / (2 * self.tones**2)
            case 'BSt_v0':
                observation = np.array([up1h, up2h, lo1h, lo2h])
                reward = ((self.tones**2 - up1h) + (self.tones**2 - up2h) + lo1h + lo2h) / (4 * self.tones**2)
            case 'BFt_v0': # TODO: なんかおかしいので観ること！
                observation = np.array([up1h, up2h, lo1h, lo2h])
                reward = ((up1h - lo1h) + (up2h - lo2h)) / (2 * self.tones**2)

            # timeあり
            case 'USo_v0t':
                observation = np.array([up1t, up1h])
                reward = (self.tones**2 - up1h) / (self.tones**2)
            case 'USt_v0t':
                observation = np.array([up1t, up1h, up2t, up2h])
                reward = ((self.tones**2 - up1h) + (self.tones**2 - up2h)) / (2 * self.tones**2)
            case 'UFtSt_v0t':
                observation = np.array([up1t, up1h, up2t, up2h])
                reward = ((self.tones**2 - up2h) - (up1h - up2h)) / (self.tones**2)
            case 'BSa_v0t':
                observation = np.array([upah, loah])
                reward = ((self.tones**2 - upah) + loah) / (2 * self.tones**2)
            case 'BSo_v0t':
                observation = np.array([up1t, up1h, lo1t, lo1h])
                reward = ((self.tones**2 - up1h) + lo1h) / (2 * self.tones**2)
            case 'BSt_v0t':
                observation = np.array([up1t, up1h, up2t, up2h, lo1t, lo1h, lo2t, lo2h])
                reward = ((self.tones**2 - up1h) + (self.tones**2 - up2h) + lo1h + lo2h) / (4 * self.tones**2)
            case 'BFt_v0t':
                observation = np.array([up1t, up1h, up2t, up2h, lo1t, lo1h, lo2t, lo2h])
                reward = ((up1h - lo1h) + (up2h - lo2h)) / (2 * self.tones**2)

            # v1's
            case 'USo_v1':
                observation = np.array([up1h])
                reward = (self.tones+2 - up1h)
            case 'USt_v1':
                observation = np.array([up1h, up2h])
                reward = (self.tones+2 - up1h) + (self.tones+2 - up2h)
            case 'UFtSt_v1':
                observation = np.array([up1h, up2h])
                reward = (self.tones+2 - up1h) + (self.tones+2 - up2h) - (up1h - up2h)
            case 'BSo_v1':
                observation = np.array([up1h, lo1h])
                reward = (self.tones+2 - up1h) + (lo1h - self.tones-2)
            case 'BSt_v1':
                observation = np.array([up1h, up2h, lo1h, lo2h])
                reward = (self.tones+2 - up1h) + (lo1h - self.tones-2) + (self.tones+2 - up2h) + (lo2h - self.tones-2)
            case 'BFt_v1':
                observation = np.array([up1h, up2h, lo1h, lo2h])
                reward = (self.tones+2 - up1h) + (lo1h - self.tones-2) + (self.tones+2 - up2h) + (lo2h - self.tones-2) - (up1h - up2h) - (lo2h - lo1h)

            # to's / tt's
            case 'USto_v1':
                observation = np.array([[up1h, up2h]])
                reward = (self.tones+2 - up1h) + (self.tones+2 - up2h)
            case 'UFtSto_v1':
                observation = np.array([[up1h, up2h]])
                reward = (self.tones+2 - up1h) + (self.tones+2 - up2h) - (up1h - up2h)
            case 'BSto_v1':
                observation = np.array([[up1h, up2h, lo1h, lo2h]])
                reward = (self.tones+2 - up1h) + (lo1h - self.tones-2) + (self.tones+2 - up2h) + (lo2h - self.tones-2)
            case 'BStt_v1':
                observation = np.array([[up1h, up2h], [lo1h, lo2h]])
                reward = (self.tones+2 - up1h) + (lo1h - self.tones-2) + (self.tones+2 - up2h) + (lo2h - self.tones-2)
            case 'BFto_v1':
                observation = np.array([[up1h, up2h, lo1h, lo2h]])
                reward = (self.tones+2 - up1h) + (lo1h - self.tones-2) + (self.tones+2 - up2h) + (lo2h - self.tones-2) - (up1h - up2h) - (lo2h - lo1h)
            case 'BFtt_v1':
                observation = np.array([[up1h, up2h], [lo1h, lo2h]])
                reward = (self.tones+2 - up1h) + (lo1h - self.tones-2) + (self.tones+2 - up2h) + (lo2h - self.tones-2) - (up1h - up2h) - (lo2h - lo1h)

            case 'dB_v0':
                observation = self.theta_k_values.tolist()
                observation.append(self.max_papr_db)
                reward = -self.max_papr_db

            case 'MSEa_v0':
                """
                criterion = nn.MSELoss()
                torch.tensor([], requires_grad=True)
                observation = np.array([self.ep_t_array])
                reward = 
                """

        # up1h は truncated 用
        return observation, reward, up1h
