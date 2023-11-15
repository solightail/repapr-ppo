import tomllib

# 設定ファイルの読み込み
with open('tests/config.toml', 'rb') as file:
    cfg = tomllib.load(file)

dict = cfg['env']['observation']
print(dict['theta_k'])