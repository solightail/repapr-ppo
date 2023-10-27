from . import algorithms as algo
from . import calc

import numpy as np
import gym
import gym.spaces
from scipy.signal import find_peaks

class MtEnv(gym.Env):
    def __init__(self, tones: int, del_freq:float, del_time:float, amp:float,
                 init_model: str='random', manual: float=None, re_model: str='USo_v0'):
        super().__init__()

        self.tones: int = tones
        self.del_freq: float = del_freq
        self.del_time: float = del_time
        self.amp: float = amp
        self.init_model: str = init_model
        self.manual: float = manual
        self.re_model: str = re_model
        self.theta_k_values: np.ndarray[float]
        self.time_values: np.ndarray = np.arange(0.0, 1.0 + self.del_time, self.del_time)

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
        self.input_dims = (self.input_dims, )

        # 環境パラメータ
        self.max_step: int = 100#*tones
        # 1行動の大きさを指定 (値を大きくしすぎると大雑把になる)
        self.action_div: float = 0.002
        self.action_div_x: float = 1.0
        self.action_list = [-10, -1, 0, 1, 10]
        self.reward_x = 10
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
        self.ep_t_array = self._eptarr()
        observation, _, _ = self._obreward(self.ep_t_array)

        return observation, None


    def step(self, action):
        # --- exec action ---
        # action を各トーンごとに分解
        each_action_tuple = np.unravel_index(action, (len(self.action_list),) * self.tones)
        for i in range(self.tones):
            self.theta_k_values[i] = self.theta_k_values[i] + ((self.action_arr[i][each_action_tuple[i]])*2*np.pi*self.action_div*self.action_div_x)
        self.ep_t_array = self._eptarr()

        # --- observation & reward ---
        observation, reward, max2_upper_peaks_array = self._obreward(self.ep_t_array)
        if np.all(each_action_tuple == 1):
            # actions において、1は停止となる
            reward = 0

        # terminated
        # 時間制限による終了
        self.steps += 1
        if (self.steps >= self.max_step):
            terminated = True
        else:
            terminated = False

        # truncated
        # エリア外等による強制終了
        if (max2_upper_peaks_array[1][0] >= self.tones**2 or
            self.tones+2 >= max2_upper_peaks_array[1][0]):
            truncated = True
        else:
            truncated = False

        return observation, reward*self.reward_x, terminated, truncated, None


    def render(self, mode='human', close=False):
        pass


    def _eptarr(self):
        formula = calc.Formula(self.tones, self.del_freq, self.amp)
        fepta = calc.FEPtA(formula, self.del_time)
        ep_t_array = fepta.get(self.theta_k_values)
        return ep_t_array

    # TODO: 増築を繰り返した結果、なんの値入れてるのか超わかりにくい状態なので、整理すること！
    def _obreward(self, ep_t_array):
        # ピーク値取得
        upper_peaks, _ = find_peaks(ep_t_array, distance=10, plateau_size=1)
        lower_peaks, _ = find_peaks(-ep_t_array, distance=10, plateau_size=1)

        # find_peaks は t = 0 にピークがある場合を考慮していない。
        # 1周期でこれを判断することは不可能であるため、2周期分で判断をし、1周期分に戻す作業を行う。
        two_cycle_ep_t_array = np.concatenate([ep_t_array, ep_t_array[1:]])
        two_cycle_upper_peaks, _ = find_peaks(two_cycle_ep_t_array, distance=10, plateau_size=1)
        two_cycle_lower_peaks, _ = find_peaks(-two_cycle_ep_t_array, distance=10, plateau_size=1)
        period: int = len(ep_t_array)-1
        if (period in two_cycle_upper_peaks):
            upper_peaks = np.concatenate([[0], upper_peaks, [period]])
        if (period in two_cycle_lower_peaks):
            lower_peaks = np.concatenate([[0], lower_peaks, [period]])

        # find_peaks 後処理
        # 最大ピークおよび準最大ピークの検出
        upper_peaks_heights = np.take(ep_t_array, upper_peaks)
        max2_reverse_upper_peaks = upper_peaks[np.argsort(upper_peaks_heights)][-2:]
        lower_peaks_heights = np.take(ep_t_array, lower_peaks)
        max2_reverse_lower_peaks = lower_peaks[np.argsort(lower_peaks_heights)][:2]

        # observation 候補
        #all_upper_peaks_array = np.array([np.take(self.time_values, upper_peaks), upper_peaks_heights])
        #all_lower_peaks_array = np.array([np.take(self.time_values, lower_peaks), lower_peaks_heights])
        #all_both_peaks_array = np.array([np.take(self.time_values, upper_peaks), upper_peaks_heights, np.take(self.time_values, lower_peaks), lower_peaks_heights])
        max2_upper_peaks_array = np.array([np.take(self.time_values, max2_reverse_upper_peaks[::-1]), np.take(ep_t_array, max2_reverse_upper_peaks[::-1])])
        max2_lower_peaks_array = np.array([np.take(self.time_values, max2_reverse_lower_peaks), np.take(ep_t_array, max2_reverse_lower_peaks)])
        max2_both_peaks_array = np.array([np.take(self.time_values, max2_reverse_upper_peaks[::-1]), np.take(ep_t_array, max2_reverse_upper_peaks[::-1]),\
                                           np.take(self.time_values, max2_reverse_lower_peaks), np.take(ep_t_array, max2_reverse_lower_peaks)])
        max_upper_peaks_array = np.array([max2_upper_peaks_array[0][0], max2_upper_peaks_array[1][0]])
        max_lower_peaks_array = np.array([max2_lower_peaks_array[0][0], max2_lower_peaks_array[1][0]])
        max_both_peaks_array = np.array([max2_upper_peaks_array[0][0], max2_upper_peaks_array[1][0], max2_lower_peaks_array[0][0], max2_lower_peaks_array[1][0]])

        # モデル別 observation & reward の算出
        match self.re_model:
            case 'USa_v0':
                avg_upper_peaks_height = np.average(np.take(ep_t_array, upper_peaks))
                observation = np.array([avg_upper_peaks_height])
                reward = (self.tones**2 - avg_upper_peaks_height) / (self.tones**2)
            case 'USo_v0':
                observation = np.array([max_upper_peaks_array[1]])
                reward = (self.tones**2 - max_upper_peaks_array[1]) / (self.tones**2)
            case 'USt_v0':
                observation = np.array([max2_upper_peaks_array[1][0], max2_upper_peaks_array[1][1]])
                reward = ((self.tones**2 - max2_upper_peaks_array[1][0]) + (self.tones**2 - max2_upper_peaks_array[1][1])) / (2 * self.tones**2)
            case 'UFtSt_v0':
                observation = np.array([max2_upper_peaks_array[1][0], max2_upper_peaks_array[1][1]])
                reward = ((self.tones**2 - max2_upper_peaks_array[1][1]) - (max2_upper_peaks_array[1][0] - max2_upper_peaks_array[1][1])) / (self.tones**2)
            case 'BSa_v0':
                avg_upper_peaks_height = np.average(np.take(ep_t_array, upper_peaks))
                avg_lower_peaks_height = np.average(np.take(ep_t_array, lower_peaks))
                observation = np.array([avg_upper_peaks_height, avg_lower_peaks_height])
                reward = ((self.tones**2 - avg_upper_peaks_height) + avg_lower_peaks_height) / (2 * self.tones**2)
            case 'BSo_v0':
                observation = np.array([max_upper_peaks_array[1], max_lower_peaks_array[1]])
                reward = ((self.tones**2 - max_upper_peaks_array[1]) + max_lower_peaks_array[1]) / (2 * self.tones**2)
            case 'BSt_v0':
                observation = np.array([max2_upper_peaks_array[1][0], max2_upper_peaks_array[1][1], max2_lower_peaks_array[1][0], max2_lower_peaks_array[1][1]])
                reward = ((self.tones**2 - max2_upper_peaks_array[1][0]) + (self.tones**2 - max2_upper_peaks_array[1][1]) +\
                          max2_lower_peaks_array[1][0] + max2_lower_peaks_array[1][1]) / (4 * self.tones**2)
            case 'BFt_v0': # TODO: なんかおかしいので観ること！
                observation = np.array([max2_upper_peaks_array[1][0], max2_upper_peaks_array[1][1], max2_lower_peaks_array[1][0], max2_lower_peaks_array[1][1]])
                reward = ((max2_upper_peaks_array[1][0] - max2_lower_peaks_array[1][0]) + (max2_upper_peaks_array[1][1] - max2_lower_peaks_array[1][1])) / (2 * self.tones**2)
            
            # timeあり
            case 'USa_v0t':
                avg_upper_peaks_height = np.average(np.take(ep_t_array, upper_peaks))
                observation = max2_upper_peaks_array
                reward = (self.tones**2 - avg_upper_peaks_height) / (self.tones**2)
            case 'USo_v0t':
                observation = max_upper_peaks_array
                reward = (self.tones**2 - max_upper_peaks_array[1]) / (self.tones**2)
            case 'USt_v0t':
                observation = max2_upper_peaks_array
                reward = ((self.tones**2 - max2_upper_peaks_array[1][0]) + (self.tones**2 - max2_upper_peaks_array[1][1])) / (2 * self.tones**2)
            case 'UFtSt_v0t':
                observation = max2_upper_peaks_array
                reward = ((self.tones**2 - max2_upper_peaks_array[1][1]) - (max2_upper_peaks_array[1][0] - max2_upper_peaks_array[1][1])) / (self.tones**2)
            case 'BSa_v0t':
                avg_upper_peaks_height = np.average(np.take(ep_t_array, upper_peaks))
                avg_lower_peaks_height = np.average(np.take(ep_t_array, lower_peaks))
                observation = max2_both_peaks_array
                reward = ((self.tones**2 - avg_upper_peaks_height) + avg_lower_peaks_height) / (2 * self.tones**2)
            case 'BSo_v0t':
                observation = max_both_peaks_array
                reward = ((self.tones**2 - max_upper_peaks_array[1]) + max_lower_peaks_array[1]) / (2 * self.tones**2)
            case 'BSt_v0t':
                observation = max2_both_peaks_array
                reward = ((self.tones**2 - max2_upper_peaks_array[1][0]) + (self.tones**2 - max2_upper_peaks_array[1][1]) +\
                          max2_lower_peaks_array[1][0] + max2_lower_peaks_array[1][1]) / (4 * self.tones**2)
            case 'BFt_v0t':
                observation = max2_both_peaks_array
                reward = ((max2_upper_peaks_array[1][0] - max2_lower_peaks_array[1][0]) + (max2_upper_peaks_array[1][1] - max2_lower_peaks_array[1][1])) / (2 * self.tones**2)

        return observation, reward, max2_upper_peaks_array




class Multitone:
    """ gym-0.26.2 互換 (new_step_api=True) """
    def __init__ (self, tones: int, del_freq:float, del_time:float, amp:float, init_model: str, re_model: str='USa1_v0') -> None:
        self.tones: int = tones
        self.del_freq: float = del_freq
        self.del_time: float = del_time
        self.amp: float = amp
        self.init_model: str = init_model
        self.re_model: str = re_model
        self.theta_k_values: np.ndarray[float]
        self.time_values: np.ndarray = np.arange(0.0, 1.0 + self.del_time, self.del_time)

        match self.re_model:
            case 'USa1_v0':
                self.n_action = 2
                self.input_dims = 1
            case 'USa2_v0':
                self.n_action = 2
                self.input_dims = 2
            case 'USo_v0':
                self.n_action = 2
                self.input_dims = 2
            case 'USt_v0':
                self.n_action = 2
                self.input_dims = 2
            case 'UFtSt_v0':
                self.n_action = 2
                self.input_dims = 2
            case 'BSa1_v0':
                self.n_action = 2
                self.input_dims = 2
            case 'BSa2_v0':
                self.n_action = 2
                self.input_dims = 4
            case 'BSo_v0':
                self.n_action = 2
                self.input_dims = 4
            case 'BSt_v0':
                self.n_action = 2
                self.input_dims = 4
            case 'BFt_v0':
                self.n_action = 2
                self.input_dims = 4

    def reset(self, manual:str=None) -> tuple:
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
                strategy = algo.Manual(self.tones, manual)

        # theta_k 計算
        algo_context = AContext(strategy)
        self.theta_k_values = algo_context.calc_algo()
        return tuple(self.theta_k_values, {})

    def step(self, action) -> tuple:
        # インスタンス生成
        formula = Formula(self.tones, self.del_freq, self.amp)
        fepta = FEPtA(formula, self.del_time)

        # --- exec action ---
        self.theta_k_values = self.theta_k_values + action
        ep_t_array = fepta.get(self.theta_k_values)

        # --- observation & reward ---
        # ピーク値取得
        upper_peaks, _ = find_peaks(ep_t_array, distance=10, plateau_size=1)
        lower_peaks, _ = find_peaks(-ep_t_array, distance=10, plateau_size=1)

        # find_peaks は t = 0 にピークがある場合を考慮していない。
        # 1周期でこれを判断することは不可能であるため、2周期分で判断をし、1周期分に戻す作業を行う。
        two_cycle_ep_t_array = np.concatenate([ep_t_array, ep_t_array[1:]])
        two_cycle_upper_peaks, _ = find_peaks(two_cycle_ep_t_array, distance=10, plateau_size=1)
        two_cycle_lower_peaks, _ = find_peaks(-two_cycle_ep_t_array, distance=10, plateau_size=1)
        period: int = len(ep_t_array)-1
        if (period in two_cycle_upper_peaks):
            upper_peaks = np.concatenate([[0], upper_peaks, [period]])
        if (period in two_cycle_lower_peaks):
            lower_peaks = np.concatenate([[0], lower_peaks, [period]])

        # find_peaks 後処理
        # 最大ピークおよび準最大ピークの検出
        upper_peaks_heights = np.take(ep_t_array, upper_peaks)
        max2_reverse_upper_peaks = upper_peaks[np.argsort(upper_peaks_heights)][-2:]
        lower_peaks_heights = np.take(ep_t_array, lower_peaks)
        max2_reverse_lower_peaks = lower_peaks[np.argsort(lower_peaks_heights)][:2]

        # observation 候補
        all_upper_peaks_array = np.ndarray(np.take(self.time_values, upper_peaks), upper_peaks_heights)
        all_lower_peaks_array = np.ndarray(np.take(self.time_values, lower_peaks), lower_peaks_heights)
        all_both_peaks_array = np.ndarray(np.take(self.time_values, upper_peaks), upper_peaks_heights, np.take(self.time_values, lower_peaks), lower_peaks_heights)
        max2_upper_peaks_array = np.ndarray(np.take(self.time_values, max2_reverse_upper_peaks[::-1]), np.take(ep_t_array, max2_reverse_upper_peaks[::-1]))
        max2_lower_peaks_array = np.ndarray(np.take(self.time_values, max2_reverse_lower_peaks), np.take(ep_t_array, max2_reverse_lower_peaks))
        max2_both_peaks_array = np.ndarray(np.take(self.time_values, max2_reverse_upper_peaks[::-1]), np.take(ep_t_array, max2_reverse_upper_peaks[::-1]),\
                                           np.take(self.time_values, max2_reverse_lower_peaks), np.take(ep_t_array, max2_reverse_lower_peaks))
        max_upper_peaks_array = np.ndarray(max2_upper_peaks_array[0][0], max2_upper_peaks_array[1][0])
        max_lower_peaks_array = np.ndarray(max2_lower_peaks_array[0][0], max2_lower_peaks_array[1][0])
        max_both_peaks_array = np.ndarray(max2_upper_peaks_array[0][0], max2_upper_peaks_array[1][0], max2_lower_peaks_array[0][0], max2_lower_peaks_array[1][0])

        # モデル別 observation & reward の算出
        match self.re_model:
            case 'USa1_v0':
                avg_upper_peaks_height = np.average(np.take(ep_t_array, upper_peaks))
                observation = np.ndarray(avg_upper_peaks_height)
                reward = (self.tones**2 - avg_upper_peaks_height) / (self.tones**2)
            case 'USa2_v0':
                avg_upper_peaks_height = np.average(np.take(ep_t_array, upper_peaks))
                observation = max2_upper_peaks_array
                reward = (self.tones**2 - avg_upper_peaks_height) / (self.tones**2)
            case 'USo_v0':
                observation = max_upper_peaks_array
                reward = (self.tones**2 - max_upper_peaks_array[1]) / (self.tones**2)
            case 'USt_v0':
                observation = max2_upper_peaks_array
                reward = ((self.tones**2 - max2_upper_peaks_array[1][0]) + (self.tones**2 - max2_upper_peaks_array[1][1])) / (2 * self.tones**2)
            case 'UFtSt_v0':
                observation = max2_upper_peaks_array
                reward = ((self.tones**2 - max2_upper_peaks_array[1][1]) - (max2_upper_peaks_array[1][0] - max2_upper_peaks_array[1][1])) / (self.tones**2)
            case 'BSa1_v0':
                avg_upper_peaks_height = np.average(np.take(ep_t_array, upper_peaks))
                avg_lower_peaks_height = np.average(np.take(ep_t_array, lower_peaks))
                observation = np.ndarray(avg_upper_peaks_height, avg_lower_peaks_height)
                reward = ((self.tones**2 - avg_upper_peaks_height) + avg_lower_peaks_height) / (2 * self.tones**2)
            case 'BSa2_v0':
                avg_upper_peaks_height = np.average(np.take(ep_t_array, upper_peaks))
                avg_lower_peaks_height = np.average(np.take(ep_t_array, lower_peaks))
                observation = max2_both_peaks_array
                reward = ((self.tones**2 - avg_upper_peaks_height) + avg_lower_peaks_height) / (2 * self.tones**2)
            case 'BSo_v0':
                observation = max_both_peaks_array
                reward = ((self.tones**2 - max_upper_peaks_array[1]) + max_lower_peaks_array[1]) / (2 * self.tones**2)
            case 'BSt_v0':
                observation = max2_both_peaks_array
                reward = ((self.tones**2 - max2_upper_peaks_array[1][0]) + (self.tones**2 - max2_upper_peaks_array[1][1]) +\
                          max2_lower_peaks_array[1][0] + max2_lower_peaks_array[1][1]) / (4 * self.tones**2)
            case 'BFt_v0':
                observation = max2_both_peaks_array
                reward = ((max2_upper_peaks_array[1][0] - max2_lower_peaks_array[1][0]) + (max2_upper_peaks_array[1][1] - max2_lower_peaks_array[1][1])) / (2 * self.tones**2)

        # terminated
        # 時間制限による終了
        if (observation ):
            terminated = True
        else:
            terminated = False

        # truncated
        # エリア外等による強制終了
        if (self.tones**2 <= observation <= self.tones+1):
            truncated = True
        else:
            truncated = False

        return observation, reward, terminated, truncated, _
