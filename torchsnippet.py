import torch
import time
import numpy as np
import os, shutil
import torch.nn as nn
import sklearn, threading
from config import Configuration as cfg
from logger import Logger



class NNprepare(object):
    """
    This class contains necessary method to build model.
    """
    def __init__(self, main_path=None, save_file_name=None, custom_NN=None):
        self.main_path = main_path
        self.save_file_name = save_file_name
        self.custom_NN = custom_NN

    def _create_save_file_name(self):
        """
        The template of name could be as follow.
        from config import Configuration as cfg
        if cfg.pretrain_solver == 'sgd':
            name = 'normal:{}_outlier:{}_{}_{}_m{}_seed{}_bs{}_epochs{}_{}'.format(cfg.pretrain_mnist_normal,
                                                                                   cfg.pretrain_mnist_outlier,
                                                                                   cfg.pretrain_solver,
                                                                                   str(cfg.pretrain_lr),
                                                                                   str(cfg.pretrain_momentum),
                                                                                   cfg.seed,
                                                                                   cfg.pretrain_batch_size,
                                                                                   cfg.pretrain_epochs,
                                                                                   self.save_file_name)
        elif cfg.pretrain_solver == 'adam':
            name = 'normal:{}_outlier:{}_{}_{}_seed{}_bs{}_epochs{}_{}'.format(cfg.pretrain_mnist_normal,
                                                                               cfg.pretrain_mnist_outlier,
                                                                               cfg.pretrain_solver,
                                                                               str(cfg.pretrain_lr),
                                                                               cfg.seed,
                                                                               cfg.pretrain_batch_size,
                                                                               cfg.pretrain_epochs,
                                                                               self.save_file_name)
        return name
        """
        raise NotImplementedError

    def _score_function(self, data=None, target=None, criterion=None, model=None):
        """
        This function is flexible yet the input and output should follow the exact same procedure.

        The following is a template of score function
        output = model(data)
        sq_loss = criterion(output, data)
        scores = torch.sum(sq_loss, dim=(1, 2, 3))
        loss = torch.mean(sq_loss)

        return output, scores, loss
        """
        raise NotImplementedError

    def _metrics(self, output, target, scores):
        raise NotImplementedError

    def make_network(self, train_loader=None, val_loader=None,
                     test_loader=None,
                     if_checkpoint_save=None,
                     print_result_epoch=None,
                     target_reshape=None,
                     print_metric_name=None,
                     model=None, **kwargs):
        if self.custom_NN is None:
            return NN(train_loader=train_loader, val_loader=val_loader,
                      test_loader=test_loader,
                      if_checkpoint_save=if_checkpoint_save,
                      print_result_epoch=print_result_epoch,
                      target_reshape=target_reshape,
                      print_metric_name=print_metric_name,
                      metrics=self._metrics,
                      create_save_file_name=self._create_save_file_name,
                      score_function=self._score_function,
                      main_path = self.main_path,
                      model=model, **kwargs)
        else:
            return self.custom_NN(
                    train_loader=train_loader, val_loader=val_loader,
                    test_loader=test_loader,
                    if_checkpoint_save=if_checkpoint_save,
                    print_result_epoch=print_result_epoch,
                    target_reshape=target_reshape,
                    print_metric_name=print_metric_name,
                    metrics=self._metrics,
                    create_save_file_name=self._create_save_file_name,
                    score_function=self._score_function,
                    main_path=self.main_path,
                    model=model, **kwargs)



class NN(object):
    """
    This is a prototype for NN wrapper in Pytorch.
    Please follow this coding style carefully.
    Args:
        model:
            Pytorch Model.
        train_loader (torch.dataset.DataLoader) :
            pytorch DataLoader for training dataset
        val_loader (torch.dataset.DataLoader) :
            pytorch DataLodaer for validation dataset
        epochs:

        opt (torch.optim) :
            optimizer
        criterion:
            Loss function.
        initial_lr (float):
            Initial learning rate. TODO implement using lr_find()
        checkpoint_save (str):
            Directory to save check point.
        model_save (str):
            Directory to save model.
        dataset:

        model:
            Pytorch Model.
        param_diagonstic (bool):
            check parameters, will be print. TODO record parameters.
        if_checkpoint_save (bool):
            save checkpoint if True
        print_result_epoch (bool):
            true if results some steps at every epochs are print.
        metrics :
            Evaluation metrics.
    """

    def __init__(self, model=None, train_loader=None, val_loader=None,
                 test_loader=None,
                 if_checkpoint_save=True,
                penalty=None,
                 print_result_epoch=False,
                 print_metric_name=None,
                 metrics=None,
                 score_function=None,
                 create_save_file_name=None,
                 target_reshape=None, **kwargs):
        self.test_loader = test_loader
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.train_current_batch_data = {}
        self.valid_current_batch_data = {}
        self.if_checkpoint_save = if_checkpoint_save
        self.print_result_epoch = print_result_epoch
        self.penalty = penalty
        self.target_reshape = target_reshape
        self.metrics = metrics
        self.score_function = score_function
        self.create_save_file_name = create_save_file_name
        self.print_metric_name = print_metric_name


        self.epochs = self.model.get_epochs()
        self._optimizer = self.model.get_optimizer()
        self._criterion = self.model.get_criterion()
        self._lr_adjust = self.model.get_lr_scheduler()
        self._tensorboard_path = self.model.get_tensorboard_path()
        self._save_path = self.model.get_logger_path()
        self._logger = Logger(self._tensorboard_path)

        if not os.path.exists(os.path.join(self._save_path, 'train_save')):
            os.makedirs(os.path.join(self._save_path, 'train_save'))
        if not os.path.exists(os.path.join(self._save_path, 'test_save')):
            os.makedirs(os.path.join(self._save_path, 'test_save'))
        print(self._save_path, self.create_save_file_name())

        self.train_checkpoint_save = os.path.join(self._save_path, 'train_save', self.create_save_file_name() + '_ckpt.path.tar')
        self.train_model_save = os.path.join(self._save_path,'train_save', self.create_save_file_name() + '_best.path.tar')

        self.test_checkpoint_save = os.path.join(self._save_path, 'test_save',
                                                 self.create_save_file_name() + '_ckpt.path.tar')
        self.test_model_save = os.path.join(self._save_path, 'test_save',
                                             self.create_save_file_name() + '_best.path.tar')

        if not isinstance(self._optimizer, torch.optim.Optimizer):
            raise TypeError('should be an torch.optim.Optimizer type, instead of {}'.format(type(self._optimizer)))

        global best_val_acc, best_test_acc

    def train(self):
        print('Start training process.')
        self.adjust_learning_rate()


        best_val_acc = 0
        best_test_acc = 0

        for epoch in range(self.epochs):
            self.train_epoch()
            if not torch.cuda.is_available() and self.test_loader is not None:
                self.multi_threading_val_test()
            elif torch.cuda.is_available() and self.test_loader is not None:
                self.evaluate()
                self.validate_epoch()
            else:
                self.validate_epoch()
            info = {'train_loss': self._train_loss.avg, 'train_{}'.format(self.print_metric_name): self._train_score.avg,
                    'val_loss': self._valid_loss.avg, 'val_{}'.format(self.print_metric_name): self._valid_score.avg}


            if self.if_checkpoint_save and self.test_loader is None:
                is_best = self._valid_score.avg > best_val_acc
                if is_best:
                    self.set_best_valid_score(self._valid_score.avg)
                print('>>>>>>>>>>>>>>>>>>>>>>')
                print(
                    'Epoch: {} train loss: {}, train {}: {}, valid loss: {}, valid {}: {}'.format(epoch, self._train_loss.avg,
                                                                                                  self._train_score.avg,
                                                                                                  self.print_metric_name,
                                                                                                  self._valid_loss.avg,
                                                                                                  self.print_metric_name,
                                                                                                  self._valid_score.avg))
                print('>>>>>>>>>>>>>>>>>>>>>>')
                self.save_checkpoint({'epoch': epoch + 1,
                                      'state_dict': self.model.state_dict(),
                                      'best_val_acc': best_val_acc,
                                      'optimizer': self._optimizer.state_dict(), }, is_best,
                                     self.train_checkpoint_save, self.train_model_save)
                info['best_val_{}'.format(self.print_metric_name)]= self.model._best_valid_score

            elif self.if_checkpoint_save and self.test_loader is not None:
                is_best = self._valid_score.avg > best_val_acc
                if is_best:
                    best_val_acc = self._valid_score.avg
                    self.set_best_valid_score(self._valid_score.avg)
                print('>>>>>>>>>>>>>>>>>>>>>>')
                print(
                    'Epoch: {} train loss: {}, train {}: {}, valid loss: {}, valid {}: {}'.format(epoch,
                                                                                                  self._train_loss.avg,
                                                                                                  self._train_score.avg,
                                                                                                  self.print_metric_name,
                                                                                                  self._valid_loss.avg,
                                                                                                  self.print_metric_name,
                                                                                                  self._valid_score.avg))
                print('>>>>>>>>>>>>>>>>>>>>>>')
                self.save_checkpoint({'epoch': epoch + 1,
                                      'state_dict': self.model.state_dict(),
                                      'best_val_acc': best_val_acc,
                                      'optimizer': self._optimizer.state_dict(), }, is_best,
                                     self.train_checkpoint_save, self.train_model_save)

                is_best_test = self._test_score.avg > best_test_acc
                if is_best_test:
                    best_test_acc = self._test_score.avg
                    self.set_best_test_score(self._test_score.avg)
                self.save_checkpoint({'epoch': epoch + 1,
                                      'state_dict': self.model.state_dict(),
                                      'best_test_acc': best_test_acc,
                                      'optimizer': self._optimizer.state_dict(), }, is_best_test,
                                     self.test_checkpoint_save, self.test_model_save)

                info['best_val_{}'.format(self.print_metric_name)] = self._best_valid_score
                info['best_test_{}'.format(self.print_metric_name)]=self._best_test_score
                info['test_{}'.format(self.print_metric_name)] = self._test_score.avg

            for tag, value in info.items():
                self._logger.scalar_summary(tag, value, epoch+1)

        print('Training process end.')

    def train_epoch(self, print_freq=100):
        """
        Train function for every epoch. Standard for supervised learning.
        Args:
            print_freq(int): number of step to print results. The first round always print.
        """
        losses = self.AverageMeter()
        percent_acc = self.AverageMeter()
        self.model.train()
        time_now = time.time()

        for batch_idx, (data, target) in enumerate(self.train_loader):
            target = target.float()

            if self.target_reshape is not None:
                target = self.target_reshape(target)
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()

            if self.score_function is None:
                output = self.model(data)
                loss = self._criterion(output, target)
            else:
                output, scores, loss = self.score_function(data, target, self._criterion, self.model)

            if self.penalty is not None:
                penalty_val = self.loss_penalty()
                loss += penalty_val

            losses.update(loss.item(), data.size(0))

            if torch.cuda.is_available():
                target = target.to(torch.device("cpu"))
                output = output.to(torch.device("cpu"))
                if self.score_function is not None:
                    scores = scores.to(torch.device("cpu"))

            if self.score_function is None:
                acc = self.metrics(output, target)
            else:
                acc = self.metrics(output, target, scores)

            # this is design particularly for sklear.metrics.roc_auc_score
            # extreme value will occur when only one class presented in mini-batch
            if acc == 0 or acc == 1:
                acc = percent_acc.avg

            percent_acc.update(acc, data.size(0))

            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

            time_end = time.time() - time_now
            #if batch_idx % print_freq == 0 and self.print_result_epoch:
            print('Training Round: {}, Time: {}'.format(batch_idx,
                                                        np.round(time_end, 2)))
            print('Training Loss: val:{} avg:{} {}: val:{} avg:{}'.format(losses.val,
                                                                          losses.avg,
                                                                          self.print_metric_name,
                                                                          percent_acc.val, percent_acc.avg))
        self.set_train_loss(losses)
        self.set_train_score(percent_acc)

    def validate_epoch(self, print_freq=10000):
        """
        Validation function for every epoch.
        Args:
            print_freq(int): number of step to print results. The first round always print.
        """
        self.model.eval()
        losses = self.AverageMeter()
        percent_acc = self.AverageMeter()

        with torch.no_grad():
            time_now = time.time()
            for batch_idx, (data, target) in enumerate(self.val_loader):
                if self.target_reshape is not None:
                    target = self.target_reshape(target)
                target = target.float()
                if torch.cuda.is_available():
                    data = data.cuda()
                    target = target.cuda()

                if self.score_function is None:
                    output = self.model(data)
                    loss = self._criterion(output, target)
                else:
                    output, scores, loss = self.score_function(data, target, self._criterion, self.model)

                if self.penalty is not None:
                    penalty_val = self.loss_penalty()
                    loss += penalty_val

                losses.update(loss.item(), data.size(0))

                if torch.cuda.is_available():
                    target = target.to(torch.device("cpu"))
                    output = output.to(torch.device("cpu"))
                    if self.score_function is not None:
                        scores = scores.to(torch.device("cpu"))

                if self.score_function is None:
                    acc = self.metrics(output, target)
                else:
                    acc = self.metrics(output, target, scores)

                percent_acc.update(acc, data.size(0))
                time_end = time.time() - time_now
                if batch_idx % print_freq == 0 and self.print_result_epoch:
                    print('Validation Round: {}, Time: {}'.format(batch_idx, np.round(time_end, 2)))
                    print('Validation Loss: val:{} avg:{} {}: val:{} avg:{}'.format(losses.val, losses.avg, self.print_metric_name,
                                                                                     percent_acc.val, percent_acc.avg))
        self.set_valid_score(percent_acc)
        self.set_valid_loss(losses)

    def adjust_learning_rate(self):
        if self._lr_adjust is not None:
            if not isinstance(self._lr_adjust, torch.optim.lr_scheduler._LRScheduler):
                raise TypeError('should be inheritant learning rate scheudler.')
            self._lr_adjust.step()
        else:
            print('Learning rate re-schedular is not setting')
        """
        lr = self.initial_lr - 0.0000  # reduce 10 percent every 50 epoch
        for param_group in opt.param_groups:
            param_group['lr'] = lr
        """

    def save_checkpoint(self, state, is_best_test, checkpoint_save, model_save):
        """
        save the best states.
        :param state:
        :param is_best: if the designated benchmark is the best in this epoch.
        :param ckpt_filename: the file path to save checkpoint, will be create if not exist.
        """
        #if not os.path.exists(self.checkpoint_save):
        #    os.mkdir(self.checkpoint_save)
        torch.save(state, checkpoint_save)
        if is_best_test:
            shutil.copyfile(checkpoint_save, model_save)

    def save_model(self):
        return None

    def resume_model(self, resume_file_path):
        if not os.path.exists(resume_file_path):
            raise ValueError('Resume file does not exist')
        else:
            print('=> loading checkpoint {}'.format(resume_file_path))
            checkpoint = torch.load(resume_file_path)
            start_epoch = checkpoint['epoch']
            self.best_val_acc = checkpoint['best_val_acc']
            self.model.load_state_dict(checkpoint['state_dict'])
            self._optimizer.load_state_dict(checkpoint['optimizer'])
            print('=> loaded checkpoint {} of epoch {}'.format(resume_file_path,
                checkpoint['epoch']))

    def evaluate(self, weights=None, print_freq=10000):
        """
        Validation function for every epoch.
        Args:
            data_loader (torch.utils.dataset.Dataloader): Dataloader for testing.
            print_freq(int): number of step to print results. The first round always print.
        """
        if weights is not None:
            print('Loading weights from {}'.format(weights))
            self.resume_model(weights)
            print('Weights loaded.')
        self.model.eval()
        percent_acc = self.AverageMeter()

        with torch.no_grad():
            time_now = time.time()
            for batch_idx, (data, target) in enumerate(self.test_loader):
                if torch.cuda.is_available():
                    data = data.cuda()
                    target = target.cuda()

                if self.score_function is None:
                    output = self.model(data)
                else:
                    output, scores, loss = self.score_function(data, target, self._criterion, self.model)

                if torch.cuda.is_available():
                    target = target.to(torch.device("cpu"))
                    output = output.to(torch.device("cpu"))
                    if self.score_function is not None:
                        scores = scores.to(torch.device("cpu"))

                if self.score_function is None:
                    acc = self.metrics(output, target)
                else:
                    acc = self.metrics(output, target, scores)

                percent_acc.update(acc, data.size(0))
                time_end = time.time() - time_now
        print('Test {}: val:{} avg:{}'.format(self.print_metric_name, percent_acc.val, percent_acc.avg))
        if not weights:
            print('Test evaluation is end!')
        self.set_test_score(percent_acc)

    def loss_penalty(self):
        l1_crit = nn.L1Loss(size_average=False)
        if self.penalty['type'] == 'l2':
            l2_penalty = 0

            for param in self.model.parameters():
                l2_penalty = torch.norm(param, 2) + l2_penalty
            l2_penalty = l2_penalty * (0.5 / self.penalty['reg'])
            return l2_penalty
        else:
            raise ValueError('Currently only l2 penalty are supported')

    def set_test_score(self, score):
        self._test_score = score

    def set_valid_score(self, score):
        self._valid_score = score

    def set_valid_loss(self, loss):
        self._valid_loss = loss

    def set_train_score(self, score):
        self._train_score = score

    def set_train_loss(self, loss):
        self._train_loss = loss

    def set_best_valid_score(self, score):
        self._best_valid_score = score

    def set_best_test_score(self, score):
        self._best_test_score = score

    def get_best_valid_score(self):
        try:
            return self._best_valid_score
        except Exception:
            print('best valid score is not defined')

    def get_best_test_score(self):
        try:
            return self._best_test_score
        except Exception:
            print('best test score is not defined')

    def multi_threading_val_test(self):
        """
        Multi threading mode to validation and test at the same time.
        """
        val_thread = threading.Thread(target=self.validate_epoch)
        val_thread.start()
        test_tread = threading.Thread(target=self.evaluate)
        test_tread.start()

        val_thread.join()
        test_tread.join()

    class AverageMeter(object):
        """Computes and stores the average and current value"""

        def __init__(self):
            self.reset()

        def reset(self):
            self.val = 0
            self.avg = 0
            self.sum = 0
            self.count = 0

        def update(self, val, n=1):
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count


class ExtendNNModule(nn.Module):
    """
    An extend version of nn.Module. Including some custom function that can be used to simplified
    to build module.
    """

    def __init__(self):
        super(ExtendNNModule, self).__init__()

    def forward(self, *input):
        raise NotImplementedError

    def _load_pretrain_weights(self, pretrain_weights_path):
        """
        Load pretrain weights into new model.
        WARNING: this method can only be used if current model have the exact save
        layers as the Pretrain model. TODO!!!
        Args:
            pretrain_weights_path (str) :
                weights path for pretrain model
            model (nn.Moduel):
                current model to load weights.
        """

        def _hard_weights_loader():
            """
            this method can only be used if current model have the exact save
            layers as the pretrain model.
            """
            print('Loading Pretrain Weights from {}'.format(pretrain_weights_path))
            pretrain_weights = torch.load(pretrain_weights_path)
            pretrain_state_dict = pretrain_weights['state_dict']
            model_state_dict = self.state_dict()
            pretrain_state_dict = {k: v for k, v in pretrain_state_dict.items() if k in model_state_dict}
            model_state_dict.update(pretrain_state_dict)
            self.load_state_dict(pretrain_state_dict)
            print('Pretrain Weights Finish Loading')
            return self

        if os.path.exists(pretrain_weights_path):
            return _hard_weights_loader()
        else:
            print('Warning! Model are trained from scratch! ')

    def _load_data(self, dataloader, batch_size, workers, shuffle=True, pin_memory=True, **kwargs):
        """
        Load data using DataLoader. Training, validation and test should all be included.
        Args:
            dataloader (torch.utils.Dataset):
                Custom dataloader function
            batch_size (int) :
                batch size.
            num_workers (int):
                threading number.
            shuffle (bool):
                shuffle the dataloader.
        """
        from torch.utils.data import DataLoader
        train_loader = DataLoader(dataloader(datatype='train', **kwargs),
                                  batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory,
                                  num_workers=workers)
        val_loader = DataLoader(dataloader(datatype='valid', **kwargs),
                                batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory,
                                num_workers=workers)
        test_loader = DataLoader(dataloader(datatype='test', **kwargs),
                                 batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory,
                                 num_workers=workers)
        return train_loader, val_loader, test_loader

    ########## Following are setter and getter for necessary instances. ############
    def set_criterion(self, criterion):
        """
        Setter for criterion.
        """
        self._criterion = criterion

    def set_optimizer(self, optimizer):
        """
        Setter for optimizer.
        """
        self._optimizer = optimizer

    def set_lr_scheduler(self, lr_scheduler):
        """
        Setter for learning rate scheduler.
        """
        if self._optimizer is not None:
            self._lr_scheduler = lr_scheduler(self._optimizer)
        else:
            raise ValueError('optimizer must be set first')

    def set_logger_path(self, logger_path):
        """
        Setter for saving weights and models.
        """
        if not os.path.exists(logger_path):
            os.makedirs(logger_path)
        self._logger_path = logger_path

    def set_tensorboard_path(self, tensorboard_path):
        """
        Setter for saving tensorboard files.
        """
        if not os.path.exists(tensorboard_path):
            os.makedirs(tensorboard_path)
        self._tensorboard_path = tensorboard_path

    def prepare_setting(self, epochs, optimizer, criterion, tensorboard_path, logger_path, lr_scheduler=None):
        self.set_epochs(epochs)
        self.set_optimizer(optimizer)
        self.set_criterion(criterion)
        self.set_tensorboard_path(tensorboard_path)
        self.set_logger_path(logger_path)
        if lr_scheduler is not None:
            self.set_lr_scheduler(lr_scheduler)

    def set_epochs(self, epochs):
        self._epochs = epochs

    def get_epochs(self):
        return self._epochs

    def get_criterion(self):
        """
        Getter for criterion.
        """
        try:
            return self._criterion
        except Exception:
            raise ValueError('criterion has not been set')

    def get_optimizer(self):
        """
        Getter for optimizer.
        """
        try:
            return self._optimizer
        except Exception:
            raise ValueError('optimizer has not been set')

    def get_lr_scheduler(self):
        """
        Getter for learning rate scheduler.
        """
        try:
            return self._lr_scheduler
        except Exception:
            print('lr_scheduler has not been set')
            pass

    def get_logger_path(self):
        try:
            return self._logger_path
        except Exception:
            raise ValueError('logger path has not been set')

    def get_tensorboard_path(self):
        try:
            return self._tensorboard_path
        except Exception:
            raise ValueError('tensorboard path has not been set')