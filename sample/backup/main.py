import gym
import numpy as np
from backup.ppo_torch import Agent
from utils import plot_learning_curve

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    N = 20
    batch_size = 5
    n_epochs = 4
    alpha = 0.0003

    # デバッグ用
    '''
    actionspace = env.action_space
    n_actions=env.action_space.n
    print(n_actions)
    input_dims=env.observation_space.shape
    print(input_dims)
    observation = env.reset()
    '''

    # エージェント インスタンス作成
    agent = Agent(n_actions=env.action_space.n, batch_size=batch_size,
                  alpha=alpha, n_epochs=n_epochs, input_dims=env.observation_space.shape)

    # ゲーム回数
    n_games = 300
    # グラフ保存先
    figure_file = 'plots/cartpole.png'

    # ベストスコア（default: -inf）
    best_score = env.reward_range[0]
    # スコア記録
    score_history = []

    # 学習数
    learn_iters = 0
    # 平均スコア
    avg_score = 0
    # 行動回数
    n_steps = 0

    # バージョン確認
    agent.printversion()

    for i in range(n_games):
        ''' ゲーム開始 前処理 '''
        # 状態の初期化
        observation, _ = env.reset()
        done = False
        score = 0

        ''' ゲーム処理（200回ループ） '''
        while not done:
            # 状態を引数にして、エージェントが次にする行動を変数に代入
            action, prob, val = agent.choose_action(observation)
            # 行動を実行し、状態やリワード（1bit）を変数に代入
            observation_, reward, done, info = env.step(action)
            # 行動回数のカウント
            n_steps += 1
            # スコアの取得（リワードの総和がスコアとなる）
            score += reward
            # 実行した状態や行動などを全て記録
            agent.remember(observation, action, prob, val, reward, done)
            # 行動をN回（default: 20）した時に、学習を1度行う
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            # 行動後の状態を行動前の状態に代入（ループ前処理）
            observation = observation_

        ''' ゲーム後処理 '''
        # スコアを記録リストへ追加
        score_history.append(score)
        # 平均スコアの算出（過去100回）
        avg_score = np.mean(score_history[-100:])

        # 今回のゲームが終わった時点の平均スコアが、ベストスコア（過去の平均スコア）よりも高い場合
        # ベストスコアの更新と、モデルのセーブを行う
        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        # CLI 表示
        print('episode', i, 'score %.1f' % score, 'avg score %.1f' %avg_score,
              'time_steps', n_steps, 'learning_steps', learn_iters)

    ''' 後処理 '''
    # ナニコレ
    x = [i+1 for i in range(len(score_history))]
    # グラフ 表示
    plot_learning_curve(x, score_history, figure_file)
