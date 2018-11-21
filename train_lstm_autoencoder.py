import argparse, os
import torch
from config import Configuration as cfg
from torch.utils.data import DataLoader
from kpi_dataloader import SingleWindowUnsupervisedKPIDataLoader
from prepare import PrepareLSTMAutoEncoder, PrepareWaveGlow
from lstm_ae import LSTMAutoEncoder
import torch.backends.cudnn as cudnn
from skorch.callbacks.lr_scheduler import WarmRestartLR
import sys
from torch.optim import SGD, Adam
from loss import SquareErrorLoss
import parser

from wg_modules import WaveGlow
from loss import WaveGlowLoss


MAIN_PATH = os.getcwd()
LOG_PATH = os.path.join(MAIN_PATH, 'log')


def main():
    cudnn.benchmark = True

    #args = parser.parse_args()


    print('Options:')
    for (key, value) in vars(cfg.args).items():
        print("{:16}: {}".format(key, value))


    torch.manual_seed(cfg.seed)
    ############# Model pretrain ################
        #torch.manual_seed(args.pretrain_seed)
        # Pretrain DataLoader prepare.

    if cfg.model == 'lstmae':
        lstm_ae_logger_path = os.path.join(MAIN_PATH, 'log', 'lstm_ae')

        lstmae_model = LSTMAutoEncoder(input_size=1, hidden_size=cfg.hidden_size,
                                       window_size=cfg.window_size, bidirectional=cfg.bidirection)
        if torch.cuda.is_available():
            print('LSTM_Auto_encoder model is running on GPU')
            lstmae_model = lstmae_model.cuda()
            torch.cuda.manual_seed(cfg.seed)

        print('LSTM_Auto_encoder Start Loading Data')
        train_loader, val_loader, test_loader = lstmae_model._load_data(SingleWindowUnsupervisedKPIDataLoader,
                                           cfg.batch_size, 1,
                                           True, True,
                                           train_datapath=os.path.join(MAIN_PATH, 'data', 'train'),
                                            test_datapath=os.path.join(MAIN_PATH, 'data', 'test'),
                                            window_size=cfg.window_size, window_gap=1)
            # _mnist_dataload(args.pretrain_mnist_normal,args.pretrain_mnist_outlier,args.pretrain_batch_size,args.workers)
        print('LSTM_Auto_encoder Finish Loading Data')

        if cfg.optim == 'sgd':
            optim = SGD(params=lstmae_model.parameters(), lr=cfg.lr, momentum=cfg.momentum, nesterov=True)
        else:
            optim = Adam(params=lstmae_model.parameters(), lr=cfg.lr)

        lstmae_model.prepare_setting(epochs=300,
                                    optimizer=optim,
                                    criterion=SquareErrorLoss(),
                                    lr_scheduler=WarmRestartLR,
                                    tensorboard_path=os.path.join(MAIN_PATH, 'tensorlog', 'lstm_ae'),
                                    logger_path=lstm_ae_logger_path)

        prepare_lstm_ae_net = PrepareLSTMAutoEncoder().make_network(model=lstmae_model, train_loader=train_loader,
                                                               val_loader=val_loader, #test_loader=test_loader,
                                                                 print_result_epoch=False, if_checkpoint_save=True,
                                                               print_metric_name='AUC')

        prepare_lstm_ae_net.train()
    else:
        waveglow_logger_path = os.path.join(MAIN_PATH, 'log', 'waveglow')
        waveglow_model = 1
        #waveglow_model = WaveGlow()
        if torch.cuda.is_available():
            print('Pretrain model is running on GPU')
            waveglow_model = waveglow_model.cuda()
            torch.cuda.manual_seed(cfg.seed)

        print('Waveglow Start Loading Data')
        train_loader, val_loader = waveglow_model._load_data(SingleWindowUnsupervisedKPIDataLoader,
                                                                        cfg.batch_size, 1, False,
                                                                        True, True,
                                                                        train_datapath=os.path.join(MAIN_PATH, 'data',
                                                                                                    'train'),
                                                                        test_datapath=os.path.join(MAIN_PATH, 'data',
                                                                                                   'test'),
                                                                        window_size=cfg.window_size, window_gap=1)
        # _mnist_dataload(args.pretrain_mnist_normal,args.pretrain_mnist_outlier,args.pretrain_batch_size,args.workers)
        print('Waveglow Finish Loading Data')


        optim = Adam(params=waveglow_model.parameters(), lr=1e-4)

        waveglow_model.prepare_setting(epochs=300,
                                     optimizer=optim,
                                     criterion=WaveGlowLoss(),
                                     lr_scheduler=WarmRestartLR,
                                     tensorboard_path=os.path.join(MAIN_PATH, 'tensorlog', 'waveglow'),
                                     logger_path=waveglow_model)

        waveglow_net = PrepareWaveGlow().make_network(model=waveglow_model, train_loader=train_loader,
                                                                    val_loader=val_loader,  # test_loader=test_loader,
                                                                    print_result_epoch=False, if_checkpoint_save=True,
                                                                    print_metric_name='AUC')

        waveglow_net.train()


if __name__ == '__main__':
    import numpy as np
    cfg.model = 'lstmae'
    assert cfg.model in ['lstmae', 'waveglow']
    cfg.seed = 0
    main()
    for i in range(100):
        random_seed = np.random.random_integers(0, 9999999999)
        cfg.seed = random_seed
        main()

