""" repapr-ppo's core """
import os
import numpy as np

from .modules.repapr_ppo_torch import Agent
from .modules.environments import MtEnv
from .modules.utils import rt_plot_init, rt_plot_reload,\
        close_plot, plot_learning_curve, write_csv

def program():
    """ Core Program """

    # 入力変数
    tones: int = 4
    del_freq: float = 1.0
    del_time: float = 0.0001
    amp: float = 1.0
    init_model: str = "narahashi"
    manual: list = None
    re_model: str = "USo_v1"

    # 処理関連
    load_data: bool = True
    rt_graph: bool = True

    # 環境パラメータ
    max_step: int = 20#*tones               # 1エピソードあたりのステップ上限
    action_div: float = 0.002               # 行動の大きさを指定 (値を大きくしすぎると大雑把になる)
    action_list = [-10, -1, 0, 1, 10]       # 行動の選択肢
    reward_x: float = 1.0                   # スコアを見やすくするための倍数

    # ハイパーパラメータ
    N: int = 10
    batch_size: int = 5
    n_epochs: int = 4
    alpha: float = 0.0003

    # 変数の初期化
    n_calc = 300            # 試行回数
    best_score = -np.inf    # ベストスコア（default: -inf）
    score_history = []      # スコア記録

    learn_iters = 0         # 学習数
    avg_score = 0           # 平均スコア
    n_steps = 0             # 行動回数

    # 学習データ保存フォルダ 準備
    dir_name: str = f"{init_model}-{re_model}-{max_step}-{N}-{batch_size}"
    chkpt_dir = f"repapr-ppo/out/ppo/{dir_name}"
    if not os.path.exists(chkpt_dir):
        os.mkdir(chkpt_dir)

    # 環境構築
    env = MtEnv(tones=tones, del_freq=del_freq, del_time=del_time, amp=amp,
                max_step=max_step, action_div=action_div, action_list=action_list,
                reward_x=reward_x, init_model=init_model, manual=manual, re_model=re_model)

    # エージェント インスタンス作成
    agent = Agent(n_actions=env.n_action, batch_size=batch_size, alpha=alpha,
                  n_epochs=n_epochs, input_dims=env.input_dims, chkpt_dir=chkpt_dir)

    # 学習データのロード
    if os.path.exists(f"repapr-ppo/out/ppo/{dir_name}/actor_torch_ppo") and\
       os.path.exists(f"repapr-ppo/out/ppo/{dir_name}/critic_torch_ppo") and load_data:
        agent.load_models()

    # 出力関連
    # リアルタイムグラフ表示の準備
    if rt_graph is True: lines = rt_plot_init(env.time_values, env.ep_t_array)
    # データ出力フォルダの準備
    data_dir = f"repapr-ppo/out/data/{dir_name}"
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    figure_file = f"{data_dir}/result.svg"
    csv_file = f"{data_dir}/result.csv"
    # 出力データ配列の準備
    theta_k_epi_list, act_epi_list, max_ep_t_epi_list, \
        max_papr_w_epi_list, max_papr_db_epi_list = [], [], [], [], []


    for i in range(n_calc):
        ''' エピソード 前処理 '''
        # 状態の初期化
        observation, _ = env.reset()
        if rt_graph is True: rt_plot_reload(env.time_values, env.ep_t_array, lines, "red")

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

        ''' エピソード 処理 '''
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
            # プロット更新
            if rt_graph is True: rt_plot_reload(env.time_values, env.ep_t_array, lines, "gray")
            # csv 出力用に配列へデータの追加
            act_epi_list.append(action)
            theta_k_epi_list.append(env.theta_k_values.tolist())
            max_ep_t_epi_list.append(env.max_ep_t)
            max_papr_w_epi_list.append(env.max_papr_w)
            max_papr_db_epi_list.append(env.max_papr_db)
            # 行動をN回した時に、学習を1度行う
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            # 行動後の状態を行動前の状態に代入（ループ前処理）
            observation = observation_

        ''' エピソード 後処理 '''
        # スコアを記録リストへ追加
        score_history.append(score)
        # 平均スコアの算出
        avg_score = np.mean(score_history[-50:])

        # 平均スコアが、ベストスコア（過去の平均スコア）よりも高い場合、ベストスコアの更新と、モデルのセーブを行う
        if i >= 25:
            if avg_score > best_score:
                best_score = avg_score
                agent.save_models()

        # CLI 表示
        print(f"episode {i}  score {score:.6}  avg score {avg_score:.6}  time_steps {n_steps}  learning_steps {learn_iters}")

        # csv 出力用に配列へデータの追加
        write_csv(i, score, avg_score, act_epi_list, theta_k_epi_list, max_ep_t_epi_list, 
                  max_papr_w_epi_list, max_papr_db_epi_list, n_steps, max_step, csv_file)
        # 出力データ配列の初期化
        theta_k_epi_list, act_epi_list, max_ep_t_epi_list, \
            max_papr_w_epi_list, max_papr_db_epi_list = [], [], [], [], []
        

    ''' 全エピソード終了 後処理'''
    # RealTime グラフ表示の終了
    close_plot()
    # 学習結果 グラフ保存
    x = [i+1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)
