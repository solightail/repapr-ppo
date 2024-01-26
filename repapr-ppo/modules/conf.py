import tomllib

class Conf(object):
    def __new__(cls, *args, **kargs):
        if not hasattr(cls, "_instance"):
            cls._instance = super(Conf, cls).__new__(cls)
        return cls._instance

    def __init__(self, filepath=None, *args, **karg):
        if not hasattr(self, "_init"):
            self._init = True

            # 設定ファイルの読み込み
            with open(filepath, 'rb') as file:
                cfg = tomllib.load(file)

            # 入力変数
            self.tones: int = cfg['input']['tones']
            self.del_freq: float = cfg['input']['del_freq']
            self.del_time: float = cfg['input']['del_time']
            self.amp: float = cfg['input']['amp']
            self.init_model: str = cfg['input']['init_model']
            self.manual: list = cfg['input']['manual']

            # 追加処理
            self.ignore_n_epi: bool = cfg['addproc']['ignore_n_epi']
            self.inheritance_theta_k: bool = cfg['addproc']['inheritance_theta_k']
            self.inheritance_reset: bool = cfg['addproc']['inheritance_reset']
            self.shrink_action_div: bool = cfg['addproc']['shrink_action_div']
            self.load_data: bool = cfg['addproc']['load_data']
            self.overwrite: bool = cfg['addproc']['overwrite']
            self.rt_graph: bool = cfg['addproc']['rt_graph']
            self.notify: bool = cfg['addproc']['notify']

            # 観測・報酬パラメータ
            self.observation_items: dict = cfg['env']['observation']
            self.eval_metrics: str = cfg['env']['reward']['eval_metrics']
            self.eval_model: str = cfg['env']['reward']['eval_model']

            # 環境パラメータ
            self.n_calc: int = cfg['env']['param']['n_calc']
            self.n_inherit: int = cfg['env']['param']['n_inherit']
            self.max_step: int = cfg['env']['param']['max_step'] #* tones
            self.action_div: float = cfg['env']['param']['action_div']
            self.action_div_shrink_scale: float = cfg['env']['param']['action_div_shrink_scale']
            self.action_list: list = cfg['env']['param']['action_list']

            # ハイパーパラメータ
            self.N: int = cfg['hyper']['N']
            self.batch_size: int = cfg['hyper']['batch_size']
            self.n_epochs: int = cfg['hyper']['n_epochs']
            self.alpha: float = cfg['hyper']['alpha']

            # スコア処理パラメータ
            self.score_avg_init: int = cfg['score']['avg_init']
            self.score_avg_calc: int = cfg['score']['avg_calc']

            # 出力
            self.filepath: str = cfg['output']['filepath']
            # LINE設定
            self.line: dict = cfg['output']['line']
