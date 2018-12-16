from src.torchsnippet import NNprepare
from config import Configuration as cfg
from torch.nn.modules.loss import MSELoss, BCELoss
import torch
from src.metrics import _roc_auc_score


def create_setting_pretraindcae(model=None):
    """
    This part is fixed for pretrain DCAE for mnist from paper Deep one-class classification setting.
    adam are used in paper.
    """
    setting = {}
    if cfg.pretrain_solver == 'adam':
        optimizer = torch.optim.Adam(params=model.parameters(), lr=cfg.pretrain_lr)
    elif cfg.pretrain_solver == 'sgd':
        optimizer = torch.optim.SGD(lr=cfg.pretrain_lr, momentum=cfg.pretrain_momentum, nesterov=True, params=model.parameters())
    else:
        raise ValueError('invalid pretrain solver for using: {}'.format(cfg.pretrain_solver))
    setting['optim'] = optimizer

    if cfg.ae_loss == 'l2':
        #from loss import MSEReconstructionError
        #loss = MSEReconstructionError()
        print('using MSE')
        loss = MSELoss(reduction='none')
    if cfg.ae_loss == 'ce':
        loss = BCELoss()

    setting['criterion'] = loss
    return setting


class PrepareLSTMAutoEncoder(NNprepare):
    """
    Pretrain DCAE design for mnist and cifar10.
    """
    def __init__(self, save_file_name=None):
        super(PrepareLSTMAutoEncoder, self).__init__()
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
        output = model(data)
        ### reconstruction error
        sq_loss = criterion(output, data)
        #print('sq_loss.shape: ', sq_loss.shape)
        loss = torch.mean(sq_loss)
        #print('sq_loss ', sq_loss)

        scores = sq_loss
        #scores_ori = scores
        scores[scores <= cfg.ano_thresh] = 0
        scores[scores > cfg.ano_thresh] = 1
        print(scores)

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