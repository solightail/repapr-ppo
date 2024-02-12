import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.terminateds = []
        self.truncateds = []

        self.batch_size = batch_size

    def generate_batches(self):
        # 学習時の既定行動回数 | -> 20
        n_states = len(self.states)
        # バッチサイズを使って均等配分 | arange(0, 20, 5) -> [0, 5, 10, 15]
        batch_start = np.arange(0, n_states, self.batch_size)
        # 既定行動回数までの連番配列の作成 | -> [0, 1, 2, ... 18, 19]
        indices = np.arange(n_states, dtype=np.int64)
        # 配列の順番をランダムに変更 | -> [ 4, 19, ... 2, 9]
        np.random.shuffle(indices)
        # indicesを各配列長がバッチサイズとなるように分割 | -> [4, ... 14], ... [7, ... 9]
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states), \
                np.array(self.actions), \
                np.array(self.probs), \
                np.array(self.vals), \
                np.array(self.rewards), \
                np.array(self.terminateds), \
                np.array(self.truncateds), \
                batches

    def store_memory(self, state, action, probs, vals, reward, terminated, truncated):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.terminateds.append(terminated)
        self.truncateds.append(truncated)

    def clear_memomry(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.terminateds = []
        self.truncateds = []

class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha, fc1_dims=256, fc2_dims=256, chkpt_dir='repapr-ppo/out/ppo'):
        super(ActorNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo')
        self.actor = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, n_actions),
            nn.Softmax(dim=-1)
        )

        self.optimizer = optim.RAdam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        dist = self.actor(state)
        dist = Categorical(dist)

        return dist

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, fc1_dims=256, fc2_dims=256, chkpt_dir='repapr-ppo/out/ppo'):
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_ppo')
        self.critic = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1)
        )

        self.optimizer = optim.RAdam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)

        return value

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class Agent:
    def __init__(self, n_actions, input_dims, gamma=0.99, alpha=0.0003, gae_lambda=0.95, policy_clip=0.2, batch_size=64, N=2048, n_epochs=10, chkpt_dir=None):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda

        self.actor = ActorNetwork(n_actions, input_dims, alpha, chkpt_dir=chkpt_dir)
        self.critic = CriticNetwork(input_dims, alpha, chkpt_dir=chkpt_dir)
        self.memory = PPOMemory(batch_size)

        # reset 用に初期設定値を保存
        self.init_n_actions = n_actions
        self.init_input_dims = input_dims
        self.init_gamma = gamma
        self.init_alpha = alpha
        self.init_gae_lambda = gae_lambda
        self.init_policy_clip = policy_clip
        self.init_batch_size = batch_size
        self.init_n_epochs = n_epochs
        self.init_chkpt_dir = chkpt_dir

    def reset(self):
        self.gamma = self.init_gamma
        self.policy_clip = self.init_policy_clip
        self.n_epochs = self.init_n_epochs
        self.gae_lambda = self.init_gae_lambda

        self.actor = ActorNetwork(self.init_n_actions, self.init_input_dims, self.init_alpha, chkpt_dir=self.init_chkpt_dir)
        self.critic = CriticNetwork(self.init_input_dims, self.init_alpha, chkpt_dir=self.init_chkpt_dir)
        self.memory = PPOMemory(self.init_batch_size)

    def remember(self, state, action, probs, vals, reward, terminated, truncated):
        self.memory.store_memory(state, action, probs, vals, reward, terminated, truncated)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, observation):
        # 現在の状態 | observation を ndarray で入れると遅いようなので、array に変換してる
        state = T.tensor(np.array(observation), dtype=T.float).to(self.actor.device)

        # 現在の状態に対する各行動の確率
        dist = self.actor(state)
        # 現在の状態に対する各行動の期待報酬
        value = self.critic(state)
        # distで導いた確率より、各行動の行動値を決定
        action = dist.sample()

        # 行動後の状態が現在の状態と等しい確率
        probs = T.squeeze(dist.log_prob(action)).item()
        # 行動の抽出
        action = T.squeeze(action).item()
        # 行動に対する期待報酬の抽出
        value = T.squeeze(value).item()

        return action, probs, value

    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr, \
            reward_arr, terminated_arr, truncated_arr, batches = self.memory.generate_batches()

            #if(_ == 0):
                # CLI表示
                #print(f"action: {action_arr}")

            # --- アドバンテージの算出（ベクトル化が可能だと思われるが脳が足りない）---
            # 状態改善量：前行動に対して現状態の価値がどれぐらい改善されるか（改善量）を示すもの

            # 配列の初期化
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            # 各行動のアドバンテージ計算ループ
            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                # 1行動ごとのアドバンテージ計算ループ
                for k in range(t, len(reward_arr)-1):
                    # new_step_api 対応（terminatedとtruncatedを合成）
                    done = terminated_arr[k] or truncated_arr[k]
                    a_t += discount*(reward_arr[k] + self.gamma*vals_arr[k+1]*(1-int(done)) - vals_arr[k])
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t

            # 状態改善量と旧期待報酬の tensor 化
            advantage = T.tensor(advantage).to(self.actor.device)
            values = T.tensor(vals_arr).to(self.actor.device)


            # --- バッチ処理 ---
            # デフォルトでは20個の値を5個ごとに分割して計算をしている。詳細はgenerate_batches()を参照
            for batch in batches:
                # --- 必要なパラメータ配列をバッチ配列化 ---
                # 前状態のバッチ配列
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)
                # 旧方策のバッチ配列
                old_probs = T.tensor(old_prob_arr[batch]).to(self.actor.device)
                # 前行動のバッチ配列
                actions = T.tensor(action_arr[batch]).to(self.actor.device)

                # 現状態においての新方策のバッチ配列
                dist = self.actor(states)
                # 現状態においての新期待報酬のバッチ配列
                critic_value = self.critic(states)
                # 不要な次元の削減
                critic_value = T.squeeze(critic_value)


                # --- 行動損失の計算 ---
                # 新方策
                new_probs = dist.log_prob(actions)
                # 方策更新量（方策比）
                prob_ratio = new_probs.exp() / old_probs.exp()
                #prob_ratio = (new_probs - old_probs).exp()
                # 重み付き方策更新量
                # 状態改善量のバッチ配列と方策更新量（既にバッチ配列）の積
                weighted_probs = advantage[batch] * prob_ratio
                # 重み付き方策更新量の飛躍を防ぐため、クリップ
                weighted_clipped_probs = T.clamp(prob_ratio, 1-self.policy_clip, 1+self.policy_clip)*advantage[batch]
                # 行動損失
                # 重み付き方策更新量とクリップ重み付き方策更新量の各最小値の平均
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                # --- 予測誤差の算出 ---
                # 状態改善量と旧期待報酬の和
                returns = advantage[batch] + values[batch]
                # さらに新期待報酬を引き、2乗する
                critic_loss = (returns-critic_value)**2
                # 求められた予測誤差のバッチ配列より、平均値を求める
                critic_loss = critic_loss.mean()

                # --- 損失関数 ---
                # 行動損失と予測誤差(x0.5)の和より全体損失を算出
                total_loss = actor_loss + 0.5*critic_loss


                # --- ネットワークパラメータの更新 ---
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

            # 次の学習のため、メモリを初期化
            self.memory.clear_memomry()
