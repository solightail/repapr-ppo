from .conf import Conf
from . import algorithms as algo
from .calc import Formula, FEPtA

import numpy as np
import gym.spaces
import torch
import torch.nn as nn
from scipy.signal import find_peaks

class MtEnv(gym.Env):
    def __init__(self):
        super().__init__()

        cfg = Conf()
        self.tones: int = cfg.tones
        self.del_freq: float = cfg.del_freq
        self.del_time: float = cfg.del_time
        self.amp: float = cfg.amp
        self.max_step: int = cfg.max_step
        self.action_div: float = cfg.action_div
        self.action_list: list = cfg.action_list
        self.init_model: str = cfg.init_model
        self.manual: float = cfg.manual
        self.observation_items: dict = cfg.observation_items
        self.eval_metrics: str = cfg.eval_metrics
        self.eval_model: str = cfg.eval_model

        self.theta_k_values: np.array[float] = self._init_theta_k_values()
        self.time_values: np.array[float] = np.arange(0.0, 1.0 + self.del_time, self.del_time)
        self.known_peaks: bool = False
        self.obs_peaks_height: list = None
        self.mse = None

        # observation の次元設定とスペース設定の準備
        observation_low = []
        observation_high = []
        observation_mean = []
        minp0t = (1/(2*self.amp**2))*((self.tones)-(self.tones*self.amp**2))
        maxp0t = (1/(2*self.amp**2))*((self.tones**2)-(self.tones*self.amp**2))
        self.input_dims = 0
        if self.observation_items['theta_k'] is True:
            self.input_dims += self.tones
            for _ in range(self.tones):
                observation_low.append(np.min(self.theta_k_values))
                observation_high.append(np.max(self.theta_k_values))
                observation_mean.append((np.min(self.theta_k_values) + np.max(self.theta_k_values))/2)
        if self.observation_items['papr_db'] is True:
            self.input_dims += 1
            observation_low.append(10*np.log10(1+((2*minp0t)/self.tones)))
            observation_high.append(10*np.log10(1+((2*maxp0t)/self.tones)))
            observation_mean.append((10*np.log10(1+((2*minp0t)/self.tones)) + 10*np.log10(1+((2*maxp0t)/self.tones)))/2)
        if self.observation_items['action_div'] is True:
            self.input_dims += 1
            observation_low.append(0)
            observation_high.append(1)
            observation_mean.append(1/2)
        if self.observation_items['peaks_height'] is True:
            if self.eval_metrics == 'abs':
                match self.eval_model:
                    case 'USo_v1': value = 1
                    case 'USt_v1': value = 2
                    case 'UFtSt_v1': value = 2
                    case 'BSo_v1': value = 2
                    case 'BSt_v1': value = 4
                    case 'BFt_v1': value = 4
            else: value = 4
            self.input_dims += value
            for _ in range(value):
                observation_low.append(0)
                observation_high.append(np.inf)
                observation_mean.append(np.inf/2)
        if self.observation_items['len_both_peaks'] is True:
            self.input_dims += 1
            observation_low.append(0)
            observation_high.append(2*(self.tones-1))
            observation_mean.append(self.tones-1)
        if self.observation_items['reward'] is True:
            self.input_dims += 1
            observation_low.append(-np.inf)
            observation_high.append(np.inf)
            observation_mean.append(0)
        self.input_dims = (self.input_dims, )

        # action_space
        # action の配列をトーン数に応じて作成
        self.action_arr = np.array([self.action_list] * self.tones)
        # action の総数を計算
        self.n_action = len(self.action_list) ** self.tones
        self.action_space = gym.spaces.Discrete(self.n_action)
        # observation_space
        self.init_observation = np.array(observation_mean)
        self.observation_space = gym.spaces.Box(low=np.array(observation_low), high=np.array(observation_high), shape=self.init_observation.shape)
        # reward_range
        # デフォルト値は両端infなので、今回は定義しない
        #self.reward_range = np.array([0, 1])
        self.reset()


    def reset(self):
        self.steps = 0
        self.theta_k_values = self._init_theta_k_values()

        # observation 計算
        self.ep_t_array, self.max_ep_t, self.papr_w, self.papr_db = self._eptarr()
        self.known_peaks = False
        if self.observation_items['reward'] is True:
            reward = self._reward()
            observation = self._observation(reward)
        else:
            observation = self._observation()

        return observation, None
    
    def manual_reset(self, manual):
        self.steps = 0
        strategy = algo.Manual(self.tones, manual)

        # theta_k 計算
        algo_context = algo.AContext(strategy)
        self.theta_k_values = algo_context.calc_algo()

        # observation 計算
        self.ep_t_array, self.max_ep_t, self.papr_w, self.papr_db = self._eptarr()
        self.known_peaks = False
        if self.observation_items['reward'] is True:
            reward = self._reward()
            observation = self._observation(reward)
        else:
            observation = self._observation()

        return observation, None


    def step(self, action):
        # --- exec action ---
        # action を各トーンごとに分解
        each_action_tuple = np.unravel_index(action, (len(self.action_list),) * self.tones)
        for i in range(self.tones):
            self.theta_k_values[i] = self.theta_k_values[i] + ((self.action_arr[i][each_action_tuple[i]])*2*np.pi*self.action_div)
        self.ep_t_array, self.max_ep_t, self.papr_w, self.papr_db = self._eptarr()

        # --- observation & reward ---
        self.known_peaks = False
        if self.observation_items['reward'] is True:
            reward = self._reward()
            observation = self._observation(reward)
        else:
            observation = self._observation()
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
        """
        if self.known_peaks is False: self._calc_peaks()
        if (self.up1h <= self.tones+2 or self.tones**2 < self.up1h):
            truncated = True
        else:
            truncated = False
        """
        truncated = False

        return observation, reward, terminated, truncated, None


    def render(self, mode='human', close=False):
        """ utils.py に必要な要素を入れているのでパス """
        pass

    def _init_theta_k_values(self):
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
        return algo_context.calc_algo()

    def _eptarr(self):
        formula = Formula(self.tones, self.del_freq, self.amp)
        fepta = FEPtA(formula, self.del_time)
        return fepta.get_ept_papr(self.theta_k_values)


    def _reward(self):
        match self.eval_metrics:
            case 'abs':
                match self.eval_model:
                    case 'USo_v1':
                        if self.known_peaks is False: self._calc_peaks()
                        self.obs_peaks_height = [self.up1h]
                        reward = (self.tones+2 - self.up1h)
                    case 'USt_v1':
                        if self.known_peaks is False: self._calc_peaks()
                        self.obs_peaks_height = [self.up1h, self.up2h]
                        reward = (self.tones+2 - self.up1h) + (self.tones+2 - self.up2h)
                    case 'UFtSt_v1':
                        if self.known_peaks is False: self._calc_peaks()
                        self.obs_peaks_height = [self.up1h, self.up2h]
                        reward = (self.tones+2 - self.up1h) + (self.tones+2 - self.up2h) - (self.up1h - self.up2h)
                    case 'BSo_v1':
                        if self.known_peaks is False: self._calc_peaks()
                        self.obs_peaks_height = [self.up1h, self.lo1h]
                        reward = (self.tones+2 - self.up1h) + (self.lo1h - self.tones-2)
                    case 'BSt_v1':
                        if self.known_peaks is False: self._calc_peaks()
                        self.obs_peaks_height = [self.up1h, self.up2h, self.lo1h, self.lo2h]
                        reward = (self.tones+2 - self.up1h) + (self.lo1h - self.tones-2) + (self.tones+2 - self.up2h) + (self.lo2h - self.tones-2)
                    case 'BFt_v1':
                        if self.known_peaks is False: self._calc_peaks()
                        self.obs_peaks_height = [self.up1h, self.up2h, self.lo1h, self.lo2h]
                        reward = (self.tones+2 - self.up1h) + (self.lo1h - self.tones-2) + (self.tones+2 - self.up2h) + (self.lo2h - self.tones-2) - (self.up1h - self.up2h) - (self.lo2h - self.lo1h)
            case 'mse':
                match self.eval_model:
                    case 'AMSE_v0':
                        self.mse = self._calc_mse(self.ep_t_array)
                        reward = -self.mse
                    case 'BMSE_v0':
                        if self.known_peaks is False: self._calc_peaks()
                        both_peaks_heights = np.append(self.upper_peaks_heights, self.lower_peaks_heights)
                        self.mse = self._calc_mse(both_peaks_heights)
                        reward = -self.mse
        return reward

    def _observation(self, reward=None):
        observation = []
        if self.observation_items['theta_k'] is True:
            observation = self.theta_k_values.tolist()
        if self.observation_items['papr_db'] is True:
            observation.append(self.papr_db)
        if self.observation_items['action_div'] is True:
            observation.append(self.action_div)
        if self.observation_items['peaks_height'] is True:
            if self.known_peaks is True:
                if self.obs_peaks_height is not None:
                    observation += self.obs_peaks_height
                else:
                    observation += [self.up1h, self.up2h, self.lo1h, self.lo2h]
            else:
                self._calc_peaks()
                observation += [self.up1h, self.up2h, self.lo1h, self.lo2h]
        if self.observation_items['len_both_peaks'] is True:
            if self.known_peaks is False: self._calc_peaks()
            observation.append(len(self.upper_peaks_heights)+len(self.lower_peaks_heights))
        if self.observation_items['reward'] is True:
            observation.append(reward)
        return observation


    def _calc_peaks(self) -> None:
        self.known_peaks = True
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
        self.upper_peaks_heights = np.take(self.ep_t_array, upper_peaks)
        max2_reverse_upper_peaks = upper_peaks[np.argsort(self.upper_peaks_heights)][-2:]
        self.lower_peaks_heights = np.take(self.ep_t_array, lower_peaks)
        max2_reverse_lower_peaks = lower_peaks[np.argsort(self.lower_peaks_heights)][:2]

        # upper_peaks 平均・最大ピーク・準最大ピーク
        self.upah = np.average(np.take(self.ep_t_array, upper_peaks))
        self.up1t = np.take(self.time_values, max2_reverse_upper_peaks[1])
        self.up1h = np.take(self.ep_t_array, max2_reverse_upper_peaks[1])
        self.up2t = np.take(self.time_values, max2_reverse_upper_peaks[0])
        self.up2h = np.take(self.ep_t_array, max2_reverse_upper_peaks[0])

        # lower_peaks 平均・最大ピーク・準最大ピーク
        self.loah = np.average(np.take(self.ep_t_array, lower_peaks))
        self.lo1t = np.take(self.time_values, max2_reverse_lower_peaks[1])
        self.lo1h = np.take(self.ep_t_array, max2_reverse_lower_peaks[1])
        self.lo2t = np.take(self.time_values, max2_reverse_lower_peaks[0])
        self.lo2h = np.take(self.ep_t_array, max2_reverse_lower_peaks[0])

    def _calc_mse(self, input_arr):
        criterion = nn.MSELoss()
        input_tensor = torch.tensor(input_arr, requires_grad=True).double()
        target_array = np.full(len(input_arr), self.tones)
        target_tensor = torch.tensor(target_array).double()
        loss = criterion(input_tensor, target_tensor)
        loss.backward()
        mse = loss.detach().numpy().copy()
        return mse
