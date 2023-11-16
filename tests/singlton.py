class Conf():
    def __new__(cls, *args, **kargs):
        if not hasattr(cls, "_instance"):
            cls._instance = super(Conf, cls).__new__(cls)
        return cls._instance

    def __init__(self, filepath=None, *args, **karg):
        if not hasattr(self, "_init"):
            self._init = True
            print('ok')

conf = Conf('test')
conf2 = Conf('test2')