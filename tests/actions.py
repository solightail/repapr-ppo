import gym
import numpy as np

class MyEnv(gym.Env):
    def __init__(self, num_carts):
        self.num_carts = num_carts
        ACTION_NUM = 3 ** self.num_carts  # アクションの数
        self.action_space = gym.spaces.Discrete(ACTION_NUM)
        self.action_arr = np.array([[-1, 0, 1]] * self.num_carts)
        print()

    def step(self, action):
        # アクションを各カートの動作に分解
        each_action = np.unravel_index(action, (3,) * self.num_carts)
        cart_action = np.zeros(self.num_carts)
        for i in range(self.num_carts):
            # 各カートに対して動作を適用
            cart_action[i] = self.action_arr[i][each_action[i]]
        return cart_action

env = MyEnv(10)
action = 78
cart_action = env.step(action)
print(cart_action)