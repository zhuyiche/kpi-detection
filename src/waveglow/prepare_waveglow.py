from src.torchsnippet import NNprepare
from config import Configuration as cfg
from src.metrics import _roc_auc_score


class PrepareWaveGlow(NNprepare):
    """
    Pretrain DCAE design for mnist and cifar10.
    """
    def __init__(self, save_file_name=None):
        super(PrepareWaveGlow, self).__init__()
        if save_file_name is None:
            self.save_file_name = ""
        else:
            self.save_file_name = save_file_name

    def _metrics(self, output, target, scores):
        return _roc_auc_score(output, target, scores)

    def _score_function(self, data=None, target=None, criterion=None, model=None):
        """
        This function is flexible yet the input and output should follow the exact same procedure.
        """
        # this is forward step to calculate loss
        output = model(data)
        loss = criterion(output, data)
        scores = 0

        return output, scores, loss

    def _create_save_file_name(self):
        if cfg.optim == 'sgd':
            name = '{}{}_m{}_seed{}_bs{}_epochs{}{}'.format(
                                                           cfg.optim,
                                                           cfg.lr,
                                                           cfg.momentum,
                                                           cfg.seed,
                                                           cfg.batch_size,
                                                           cfg.epochs,
                                                           self.save_file_name)
        elif cfg.optim == 'adam':
            name = '{}{}_seed{}_bs{}_epochs{}{}'.format(
                                                       cfg.optim,
                                                       cfg.lr,
                                                       cfg.seed,
                                                       cfg.batch_size,
                                                       cfg.epochs,
                                                       self.save_file_name)
        return name