"""
class to do all the otherwise tedious boilerplate training/validation/testing code
when using pytorch library.
"""


class TorchTrainer:
    """
    Requires a pytorch nn model, path to datasets, problem to solve either classification/regression.
    """
    def __init__(self, model, datapath, problem, **kwargs):
        self.model, self.datapath, self.problem = model, datapath, problem
        self.regularization = kwargs.get['regularization']
        self.visualize = kwargs.get['visualize']
    
    
class History(TorchTrainer):
    def __init__(self, **kwargs)
        super(History, self).__init__()
        self.track = kwargs.get['type']


if __name__ == '__main__':
    pass