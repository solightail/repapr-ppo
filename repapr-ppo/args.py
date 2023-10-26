def param():
    ''' 入力変数 '''
    tones: int = 10
    del_freq: float = 1.0
    del_time: float = 0.0001
    amp: float = 1.0
    init_model: str = 'random'
    re_model: str = 'USa1_v0'

def env_param():
    ''' 環境パラメータ '''
    max_steps: int = 200



def hyper():
    ''' ハイパーパラメータ '''
    N: int = 20
    batch_size: int = 5
    n_epochs: int = 4
    alpha: float = 0.0003
