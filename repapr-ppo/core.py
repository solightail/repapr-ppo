""" repapr-ppo's core """
import os
import tomllib
import numpy as np
from datetime import datetime

from .modules.repapr_ppo_torch import Agent
from .modules.environments import MtEnv
from .modules.utils import rt_plot_init, rt_plot_reload_line, \
    rt_plot_reload_text_bl, rt_plot_reload_text_br, pause_plot, close_plot, \
    plot_learning_curve, write_csv, send_line, new_filename

def program():
    """ Core Program """

    # 設定ファイルの読み込み
    with open('repapr-ppo/config.toml', 'rb') as file:
        cfg = tomllib.load(file)

    # 入力変数
    tones: int = cfg['input']['tones']
    del_freq: float = cfg['input']['del_freq']
    del_time: float = cfg['input']['del_time']
    amp: float = cfg['input']['amp']
    init_model: str = cfg['input']['init_model']
    manual: list = cfg['input']['manual']

    # 追加処理
    inf_calc: bool = cfg['addproc']['inf_calc']
    inheritance_theta_k: bool = cfg['addproc']['inheritance_theta_k']
    shrink_action_div: bool = cfg['addproc']['shrink_action_div']
    load_data: bool = cfg['addproc']['load_data']
    overwrite: bool = cfg['addproc']['overwrite']
    rt_graph: bool = cfg['addproc']['rt_graph']
    notify: bool = cfg['addproc']['notify']

    # 観測・報酬パラメータ
    observation_items: dict = cfg['env']['observation']
    eval_metrics: str = cfg['env']['reward']['eval_metrics']
    eval_model: str = cfg['env']['reward']['eval_model']

    # 環境パラメータ
    n_calc: int = cfg['param']['n_calc']
    n_inherit: int = cfg['param']['n_inherit']
    max_step: int = cfg['param']['max_step'] #* tones
    action_div: float = cfg['param']['action_div']
    action_div_shrink_scale: float = cfg['param']['action_div_shrink_scale']
    action_list: list = cfg['param']['action_list']

    # ハイパーパラメータ
    N: int = cfg['hyper']['N']
    batch_size: int = cfg['hyper']['batch_size']
    n_epochs: int = cfg['hyper']['n_epochs']
    alpha: float = cfg['hyper']['alpha']

    # スコア処理パラメータ
    score_avg_init = cfg['score']['avg_init']
    score_avg_calc = cfg['score']['avg_calc']

    # LINE Token
    channel_token = cfg['line']['channel_token']
    user_id = cfg['line']['user_id']


    # 変数の初期化
    best_score = -np.inf                # ベストスコア（default: -inf）
    inheritance_theta_k_values = None   # theta_k 継承値
    score_history = []                  # スコア記録

    n_epi = 0                           # エピソード数
    index = 0                           # 周回数
    learn_iters = 0                     # 学習数
    avg_score = 0                       # 平均スコア
    n_steps = 0                         # 行動回数

    # 保存フォルダパス
    dir_name: str = f"{tones}-{init_model}-{eval_model}"#-{max_step}-{N}-{batch_size}"
    chkpt_dir = f"repapr-ppo/out/ppo/{dir_name}"
    data_dir = f"repapr-ppo/out/data/{dir_name}"

    # 学習データ保存フォルダ 準備
    if not os.path.exists(chkpt_dir):
        os.mkdir(chkpt_dir)
    else:
        if os.path.exists(f'{chkpt_dir}/actor_torch_ppo') and overwrite is False:
            raise FileExistsError("The file already exists. Make the overwrite true or use a different algorithm.")

    # 環境構築
    env = MtEnv(tones=tones, del_freq=del_freq, del_time=del_time, amp=amp,
                 max_step=max_step, action_div=action_div, action_list=action_list,
                 init_model=init_model, manual=manual, observation_items=observation_items,
                 eval_metrics=eval_metrics, eval_model=eval_model)

    # エージェント インスタンス作成
    agent = Agent(n_actions=env.n_action, batch_size=batch_size, alpha=alpha,
                  n_epochs=n_epochs, input_dims=env.input_dims, chkpt_dir=chkpt_dir)

    # 学習データのロード
    if os.path.exists(f"{chkpt_dir}/actor_torch_ppo") and\
       os.path.exists(f"{chkpt_dir}/critic_torch_ppo") and load_data:
        agent.load_models()


    # 出力関連
    # リアルタイムグラフ表示の準備
    if rt_graph is True: lines, plot_text_bl, plot_text_br = rt_plot_init(env.time_values, env.ep_t_array, env.papr_db, env.mse, env.action_div)

    # データ出力フォルダの準備
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    path_filename = new_filename(data_dir, "result")
    figure_file = f'{path_filename}.svg'
    csv_file = f'{path_filename}.csv'
    # 出力データ配列の準備
    theta_k_epi_list, act_epi_list, max_ep_t_epi_list, \
        papr_w_epi_list, papr_db_epi_list = [], [], [], [], []

    # 処理開始前 LINE 通知
    now = datetime.now().time()
    message = f"repapr-ppo / {now.isoformat(timespec='seconds')}\n{dir_name}\n処理を開始します"
    if notify is True:
        send_line(channel_token, user_id, message)


    # エピソード回数の決定
    if inf_calc is True:
        if inheritance_theta_k is True:
            limit = n_inherit
        else:
            limit = np.inf
    else:
        limit = n_calc

    while not (index >= limit):
        ''' エピソード 前処理 '''
        # 状態の初期化
        if inheritance_theta_k_values is None:
            observation, _ = env.reset()
            best_papr_db = env.papr_db
        else:
            observation, _ = env.manual_reset(inheritance_theta_k_values)
        if rt_graph is True:
            rt_plot_reload_line(lines, env.time_values, env.ep_t_array, "red")
            rt_plot_reload_text_br(plot_text_br, env.papr_db, env.mse, "red")
            pause_plot()

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
            if rt_graph is True:
                rt_plot_reload_line(lines, env.time_values, env.ep_t_array, "gray")
                rt_plot_reload_text_br(plot_text_br, env.papr_db, env.mse, "gray")
                pause_plot()
            # csv 出力用に配列へデータの追加
            act_epi_list.append(action)
            theta_k_epi_list.append(env.theta_k_values.tolist())
            max_ep_t_epi_list.append(env.max_ep_t)
            papr_w_epi_list.append(env.papr_w)
            papr_db_epi_list.append(env.papr_db)
            # 行動をN回した時に、学習を1度行う
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            # 行動後の状態を行動前の状態に代入（ループ前処理）
            observation = observation_

        ''' エピソード 後処理 '''
        # theta_k の継承
        min_papr_db_epi = np.min(papr_db_epi_list)
        if (min_papr_db_epi <= best_papr_db):
            best_papr_db = min_papr_db_epi
            if inheritance_theta_k is True:
                if shrink_action_div is True:
                    env.action_div = env.action_div*action_div_shrink_scale
                inheritance_theta_k_values = theta_k_epi_list[np.argmin(papr_db_epi_list)]
                rt_plot_reload_text_bl(plot_text_bl, index, best_papr_db, env.action_div, 'red')
            else:
                inheritance_theta_k_values = None
                rt_plot_reload_text_bl(plot_text_bl, index, best_papr_db, None, 'red')
        else:
            if inheritance_theta_k is True:
                if inf_calc is True: index -= 1
                rt_plot_reload_text_bl(plot_text_bl, index, best_papr_db, env.action_div, 'gray')
            else:
                rt_plot_reload_text_bl(plot_text_bl, index, best_papr_db, None, 'gray')

        # スコアを記録リストへ追加
        score_history.append(score)
        # 平均スコアの算出
        avg_score = np.mean(score_history[-score_avg_calc:])
        # 平均スコアが、ベストスコア（過去の平均スコア）よりも高い場合、ベストスコアの更新と、モデルのセーブを行う
        if n_epi >= score_avg_init:
            if avg_score > best_score:
                best_score = avg_score
                agent.save_models()

        # CLI 表示
        print(f"episode {n_epi}  score {score:.03f}  avg score {avg_score:.03f}  time_steps {n_steps}  learning_steps {learn_iters}")

        # csv 出力用に配列へデータの追加
        write_csv(n_epi, score, avg_score, act_epi_list, theta_k_epi_list, max_ep_t_epi_list, 
                  papr_w_epi_list, papr_db_epi_list, n_steps, max_step, csv_file)
        # 出力データ配列の初期化
        theta_k_epi_list, act_epi_list, max_ep_t_epi_list, \
            papr_w_epi_list, papr_db_epi_list = [], [], [], [], []

        n_epi += 1
        index += 1


    ''' 全エピソード終了 後処理'''
    # RealTime グラフ表示の終了
    close_plot()
    # 学習結果 グラフ保存
    x = [n_epi+1 for n_epi in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)
    # LINE 通知
    now = datetime.now().time()
    message = f"repapr-ppo / {now.isoformat(timespec='seconds')}\n{dir_name}\n全てのエピソードの演算が完了しました。出力データより解析を行ってください。"
    if notify is True:
        send_line(channel_token, user_id, message)
