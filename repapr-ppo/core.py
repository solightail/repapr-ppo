""" repapr-ppo's core """
import numpy as np
from .modules.repapr_ppo_torch import Agent
from .modules.environments import MtEnv
from .modules.utils import plot_learning_curve

def program():
    """ Core Program """
    ''' 入力変数 '''
    tones: int = 10
    del_freq: float = 1.0
    del_time: float = 0.0001
    amp: float = 1.0
    init_model: str = 'narahashi'
    re_model: str = 'USa1_v0'

    ''' 環境構築 '''
    env = MtEnv(tones=tones, del_freq=del_freq, del_time=del_time, 
                    amp=amp, init_model=init_model, re_model=re_model)

    ''' ハイパーパラメータ '''
    N = 20
    batch_size = 5
    n_epochs = 4
    alpha = 0.0003

    ''' エージェント インスタンス作成 '''
    agent = Agent(n_actions=env.n_action, batch_size=batch_size,
                  alpha=alpha, n_epochs=n_epochs, input_dims=env.input_dims)

    ''' 変数の初期化 '''
    n_calc = 300            # 試行回数
    best_score = tones**2   # ベストスコア（default: -inf）
    score_history = []      # スコア記録

    learn_iters = 0         # 学習数
    avg_score = 0           # 平均スコア
    n_steps = 0             # 行動回数

    # グラフ保存先
    figure_file = 'plots/cartpole.png'

    for i in range(n_calc):
        ''' 各試行 前処理 '''
        # 状態の初期化
        observation, _ = env.reset()

        # terminated:
        #   終端状態になったかの判定
        #   終端状態でリセットせずにstep()をするとWarningが出る
        #   今回は到達目標を定義せずに深層学習を行うため、常時 False となる
        # truncated:
        #   MDPの範囲外切り捨て条件を満たすかの判定
        #   タイムリミット or 範囲外 を検出する
        #   今回は、範囲外を想定しないため、タイムリミットが True になることを判定してもらう
        terminated, truncated = False, False
        score = 0

        ''' ゲーム処理（Loop one Episode） '''
        while not terminated and not truncated:
            # 状態を引数にして、エージェントが次にする行動を変数に代入
            # action: 現在の状態より決定した次の行動
            # prob: 行動後の状態が現在の状態と等しい確率
            # val: 行動後の報酬期待値
            action, prob, val = agent.choose_action(observation)
            # 行動を実行し、状態やリワードを変数に代入
            # 瞬時包絡線電力の計算はここで行う
            observation_, reward, terminated, truncated, _ = env.step(action)
            # 行動回数のカウント
            n_steps += 1
            # スコアの取得（リワードの総和がスコアとなる）
            score += reward
            # 実行した状態や行動などを記録
            agent.remember(observation, action, prob, val, reward, terminated, truncated)
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
