import pandas as pd
import numpy as np

# テスト配列
episode = 0
score = -53.3
avgscore = -53.3
data1 = [episode, score, avgscore]

"""
data1 = np.array([episode, score, avgscore])
theta_k_arr = np.array([0, 0, 0, 3.141592, 0, 0, 0, 3.141592, 0, 0, 0, 3.141592, 0, 0, 0, 3.141592, 0, 0, 0, 3.141592]).reshape(5, 4)
ept = np.array([16, 8, 12, 2, 0])
papr_w = np.array([10, 0, 2, 3, 0])
papr_db = np.array([8, 2, 12, 3, 0])
data2 = np.array([[theta_k_arr], ept, papr_w, papr_db]).T
"""

theta_k_arr = [[0, 0, 0, 3.141592], [0, 0, 0, 3.141592], [0, 0, 0, 3.141592], [0, 0, 0, 3.141592], [0, 0, 0, 3.141592]]
ept = [16, 8, 12, 2, 0]
papr_w = [10, 0, 2, 3, 0]
papr_db = [8, 2, 12, 3, 0]
data2 = []
for i in range(len(ept)):
    data2.append([np.nan, np.nan, np.nan, theta_k_arr[i], ept[i], papr_w[i], papr_db[i]])

col_name = ['episode', 'score', 'avg.score', 'theta_k', 'EP(t) [W]', 'PAPR [W]', 'PAPR [dB]']
s_for_fill = pd.Series(data1, index=col_name[:3])

# 初期化
#df = pd.DataFrame(columns = ['episode', 'score', 'avg.score', 'theta_k', 'EP(t) [W]', 'PAPR [W]', 'PAPR [dB]'])
df1 = pd.DataFrame({'episode': episode, 'score': score, 'avg.score': avgscore}, index=[0])
#df1 = pd.DataFrame(data1, columns=col_name[:3])
df2_index = np.arange(0, 5)
df2 = pd.DataFrame({'theta_k': theta_k_arr, 'EP(t) [W]': ept, 'PAPR [W]': papr_w, 'PAPR [dB]': papr_db}, index=df2_index)
#df = pd.DataFrame(data2, columns=col_name)
#sr1 = pd.Series(data2, name="sr1")
#sr_new_col = pd.Series(data2, col_name, name="new_col")

# データフレームに追加
df = pd.concat([df1, df2], axis=1)
#df_conc = pd.DataFrame(np.vstack([df1.values, sr1.values]), columns=df1.columns)

#df = df.fillna(s_for_fill)

print(df)

# CSVファイルに書き込み
#df.to_csv('tmp/data.csv')