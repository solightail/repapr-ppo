import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import requests

def new_filename(path, filename) -> str:
    i = 1
    while len(glob.glob(f"{path}/{filename}-{i}.*")) > 0:
        i += 1
    return f'{path}/{filename}-{i}'

def new_result_path(path, foldername) -> str:
    i = 1
    while os.path.exists(f'{path}/{foldername}-{i}'):
        i += 1
    return f'{path}/{foldername}-{i}', f'{foldername}-{i}'


def rt_plot_init(time_values, ept_values, papr_db, mse, action_div):
    lines, = plt.plot(time_values, ept_values)
    if action_div is None:
        plot_text_bl = plt.figtext(0.02, 0.02, f'0 Best PAPR: {papr_db:.04f} dB', ha='left', color='red')
    else:
        plt.subplots_adjust(bottom=10/72)
        plot_text_bl = plt.figtext(0.02, 0.02, f'0 Best PAPR: {papr_db:.04f} dB / action_div: x{action_div}', ha='left', color='red')
    if mse is None:
        plot_text_br = plt.figtext(0.98, 0.02, f'PAPR: {papr_db:.03f} dB', ha='right', color='red')
    else:
        plot_text_br = plt.figtext(0.98, 0.02, f'PAPR: {papr_db:.03f} dB / MSELoss: {mse:.03f}', ha='right', color='red')
    plt.xlabel('Time')
    plt.xlim(0, 1)
    plt.xticks([0, 0.5, 1], [0, 'T/2', 'T'])
    plt.ylabel('EP(t)')
    plt.ylim(0, )
    plt.legend()
    plt.grid(True)
    return lines, plot_text_bl, plot_text_br

def rt_plot_reload_line(lines, time_values, ept_values, setcolor):
    lines.set_data(time_values, ept_values)
    lines.set_color(setcolor)

def rt_plot_reload_text_bl(text, index, best_papr_db, action_div, setcolor):
    if action_div is None:
        text.set_text(f'{index} Best PAPR: {best_papr_db:.03f} dB')
    else:
        text.set_text(f'{index+1} Best PAPR: {best_papr_db:.03f} dB / action_div: x{action_div:.06f}')
    text.set_color(setcolor)

def rt_plot_reload_text_br(text, papr_db, mse, setcolor):
    if mse is None:
        text.set_text(f'PAPR: {papr_db:.03f} dB')
    else:
        text.set_text(f'PAPR: {papr_db:.03f} dB / MSELoss: {mse:.03f}')
    text.set_color(setcolor)

def pause_plot():
    plt.pause(.01)

def close_plot():
    plt.clf()
    plt.close()

def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)


def write_csv(epi, sco, avgsco, act_list, tk_list, mse_list, m_ept_list, m_pw_list, m_pd_list, n_steps, max_step, path):
    s_index = epi*max_step
    df1 = pd.DataFrame({
        'episode': epi,
        'score': sco,
        'avg_score': avgsco
    }, index=[s_index])
    df2 = pd.DataFrame({
        'action': act_list,
        'theta_k': tk_list,
        'MSE': mse_list,
        'EP(t) [W]': m_ept_list,
        'PAPR [W]': m_pw_list,
        'PAPR [dB]': m_pd_list
    }, index=np.arange(s_index, n_steps))
    df = pd.concat([df1, df2], axis=1)

    if (epi == 0):
        df.to_csv(path)
    else:
        df.to_csv(path, mode='a', header=False)

def send_line(channel_token, user_id, text):
    url = 'https://api.line.me/v2/bot/message/push'
    headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {channel_token}'}
    post = {'to': user_id, 'messages': [{'type': 'text', 'text': text}]}
    req = requests.post(url, headers=headers, data=json.dumps(post))

    if req.status_code != 200:
        print(req.text)
