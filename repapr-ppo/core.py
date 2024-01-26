""" repapr-ppo's core """
import os
import numpy as np
from datetime import datetime

from .modules.conf import Conf
from .modules.repapr_ppo_torch import Agent
from .modules.environments import MtEnv
from .modules.utils import rt_plot_init, rt_plot_reload_line, \
    rt_plot_reload_text_bl, rt_plot_reload_text_br, pause_plot, close_plot, \
    plot_learning_curve, write_csv, send_line, new_result_path

def program():
    """ Core Program """

    # 設定ファイルの読み込み
    conf_filepath = 'repapr-ppo/config.toml'
    cfg = Conf(conf_filepath)

    # 変数の初期化
    best_score = -np.inf                # ベストスコア（default: -inf）
    inheritance_theta_k_values = None   # theta_k 継承値
    score_history = []                  # スコア記録
    score_save = []                     # スコア記録（継承時プロット用）

    n_epi = 0                           # エピソード数
    index = 0                           # 周回数
    learn_iters = 0                     # 学習数
    avg_score = 0                       # 平均スコア
    n_steps = 0                         # 行動回数

    # 出力設定
    output_dir: str = f'{cfg.filepath}'
    dir_name: str = f'{cfg.tones}-{cfg.init_model}-{cfg.eval_model}'#-{max_step}-{N}-{batch_size}'
    chkpt_dir: str = f'{output_dir}/{dir_name}'
    result_dir, result_dir_name = new_result_path(chkpt_dir, 'result')
    figure_file = f'{result_dir}/{result_dir_name}.svg'
    csv_file = f'{result_dir}/{result_dir_name}.csv'

    # 学習モデル保存 準備
    if not os.path.exists(chkpt_dir):
        os.mkdir(chkpt_dir)
    else:
        if os.path.exists(f'{chkpt_dir}/actor_torch_ppo') and cfg.overwrite is False:
            raise FileExistsError("The file already exists. Make the overwrite true or use a different algorithm.")
    # 結果保存 準備
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)


    # 環境構築
    env = MtEnv()
    # エージェント インスタンス作成
    agent = Agent(n_actions=env.n_action, batch_size=cfg.batch_size, alpha=cfg.alpha,
                  n_epochs=cfg.n_epochs, input_dims=env.input_dims, chkpt_dir=chkpt_dir)

    # 学習データのロード
    if os.path.exists(f"{chkpt_dir}/actor_torch_ppo") and\
       os.path.exists(f"{chkpt_dir}/critic_torch_ppo") and cfg.load_data:
        agent.load_models()


    # 出力関連
    # リアルタイムグラフ表示の準備
    if cfg.rt_graph is True:
        lines, plot_text_bl, plot_text_br = rt_plot_init(env.time_values, env.ep_t_array, env.papr_db, env.mse, env.action_div)

    # 出力データ配列の準備
    theta_k_epi_list, act_epi_list, max_ep_t_epi_list, \
        papr_w_epi_list, papr_db_epi_list = [], [], [], [], []

    # 処理開始前 LINE 通知
    start = datetime.now()
    message = f"repapr-ppo / {start.time().isoformat(timespec='seconds')}\n{dir_name}\n処理を開始します"
    if cfg.notify is True:
        send_line(cfg.line['channel_token'], cfg.line['user_id'], message)


    # エピソード回数の決定
    if cfg.ignore_n_epi is True:
        epi_limit = cfg.n_calc
        if cfg.inheritance_theta_k is True:
            limit = cfg.n_inherit
        else:
            limit = np.inf
    else:
        epi_limit = cfg.n_calc


    while not (n_epi >= epi_limit or index >= limit):
        ''' エピソード 前処理 '''
        # 状態の初期化
        if inheritance_theta_k_values is None:
            observation, _ = env.reset()
            best_papr_db = env.papr_db
        else:
            observation, _ = env.manual_reset(inheritance_theta_k_values)
        if cfg.rt_graph is True:
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
            if cfg.rt_graph is True:
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
            if n_steps % cfg.N == 0:
                agent.learn()
                learn_iters += 1
            # 行動後の状態を行動前の状態に代入（ループ前処理）
            observation = observation_

        ''' エピソード 後処理 '''
        # theta_k の継承
        min_papr_db_epi = np.min(papr_db_epi_list)
        if (min_papr_db_epi <= best_papr_db):
            best_papr_db = min_papr_db_epi
            if cfg.inheritance_theta_k is True:
                if cfg.shrink_action_div is True:
                    env.action_div = env.action_div*cfg.action_div_shrink_scale
                    print(f'\n----- ReNew: {index+1} / action_div: {env.action_div:.04f} / PAPR: {best_papr_db:.04f} -----')
                else:
                    print(f'\n----- ReNew: {index+1} / PAPR: {best_papr_db:.04f} -----')
                inheritance = True
                inheritance_theta_k_values = theta_k_epi_list[np.argmin(papr_db_epi_list)]
                if cfg.rt_graph is True: rt_plot_reload_text_bl(plot_text_bl, index, best_papr_db, env.action_div, 'red')
            else:
                print(f'\n----- ReNew: {index+1} / PAPR: {best_papr_db:.04f} -----')
                if cfg.rt_graph is True: rt_plot_reload_text_bl(plot_text_bl, index, best_papr_db, None, 'red')
        else:
            if cfg.inheritance_theta_k is True:
                if cfg.ignore_n_epi is True: index -= 1
                if cfg.rt_graph is True: rt_plot_reload_text_bl(plot_text_bl, index, best_papr_db, env.action_div, 'gray')
            else:
                if cfg.rt_graph is True: rt_plot_reload_text_bl(plot_text_bl, index, best_papr_db, None, 'gray')

        # スコアを記録リストへ追加
        score_history.append(score)
        # 平均スコアの算出
        avg_score = np.mean(score_history[-cfg.score_avg_calc:])
        # 平均スコアが、ベストスコア（過去の平均スコア）よりも高い場合、ベストスコアの更新と、モデルのセーブを行う
        if cfg.inheritance_theta_k is False:
            # 継承を利用していない場合、初めはスコア更新を行わない
            if n_epi >= cfg.score_avg_init:
                if avg_score > best_score:
                    best_score = avg_score
                    agent.save_models()
        else:
            if avg_score > best_score:
                best_score = avg_score
                agent.save_models()

        # CLI 表示
        print(f"episode {n_epi}  score {score:.03f}  avg score {avg_score:.03f}  time_steps {n_steps}  learning_steps {learn_iters}")

        # csv 出力用に配列へデータの追加
        write_csv(n_epi, score, avg_score, act_epi_list, theta_k_epi_list, max_ep_t_epi_list, 
                  papr_w_epi_list, papr_db_epi_list, n_steps, cfg.max_step, csv_file)

        # 継承時 学習の初期化
        if cfg.inheritance_reset is True and inheritance is True:
            score_save += score_history
            score_history = []
            avg_score = 0
            best_score = -np.inf
            agent.reset()

        # ループ前 初期化
        n_epi += 1
        index += 1
        inheritance = False
        # 出力データ配列の初期化
        theta_k_epi_list, act_epi_list, max_ep_t_epi_list, \
            papr_w_epi_list, papr_db_epi_list = [], [], [], [], []


    ''' 全エピソード終了 後処理'''
    # RealTime グラフ表示の終了
    close_plot()
    # 学習結果 グラフ保存
    if cfg.inheritance_theta_k is False:
        x = [n_epi+1 for n_epi in range(len(score_history))]
        plot_learning_curve(x, score_history, figure_file)
    else:
        x = [n_epi+1 for n_epi in range(len(score_save))]
        plot_learning_curve(x, score_save, figure_file)
    # LINE 通知
    end = datetime.now()
    message = f"repapr-ppo / {end.time().isoformat(timespec='seconds')}\n{dir_name}\n全てのエピソードの演算が完了しました。出力データより解析を行ってください。"
    if cfg.notify is True:
        send_line(cfg.line['channel_token'], cfg.line['user_id'], message)
    # 経過時間表示
    print()
    print(f'PAPR[dB]: {best_papr_db}  経過時間:{end-start}')
