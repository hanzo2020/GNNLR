# coding=utf-8

import torch
import logging
import time
from time import time as t
from utils import utils
from utils.global_p import *
from tqdm import tqdm
import gc
import numpy as np
import copy
import os
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


class BaseRunner(object):
    @staticmethod
    def parse_runner_args(parser):
        """
        跑模型的命令行参数
        :param parser:
        :return:
        
        Command-line parameters to run the model
        :param parser:
        :return:
        """
        parser.add_argument('--load', type=int, default=0,
                            help='Whether load model and continue to train')
        parser.add_argument('--load_model_path', type=str, default='',
                            help='load model path')
        parser.add_argument('--epoch', type=int, default=200,
                            help='Number of epochs.')
        parser.add_argument('--check_epoch', type=int, default=1,
                            help='Check every epochs.')
        parser.add_argument('--early_stop', type=int, default=0,
                            help='whether to early-stop.')
        parser.add_argument('--lr', type=float, default=0.001,
                            help='Learning rate.')
        parser.add_argument('--batch_size', type=int, default=128,
                            help='Batch size during training.')
        parser.add_argument('--eval_batch_size', type=int, default=128 * 128,
                            help='Batch size during testing.')
        parser.add_argument('--dropout', type=float, default=0.2,
                            help='Dropout probability for each deep layer')
        parser.add_argument('--l2_bias', type=int, default=0,
                            help='Whether add l2 regularizer on bias.')
        parser.add_argument('--l2', type=float, default=1e-5,
                            help='Weight of l2_regularize in pytorch optimizer.')
        parser.add_argument('--l2s', type=float, default=0.0,
                            help='Weight of l2_regularize in loss.')
        parser.add_argument('--grad_clip', type=float, default=10,
                            help='clip_grad_value_ para, -1 means, no clip')
        parser.add_argument('--optimizer', type=str, default='Adam',
                            help='optimizer: GD, Adam, Adagrad')
        parser.add_argument('--metrics', type=str, default="RMSE",
                            help='metrics: RMSE, MAE, AUC, F1, Accuracy, Precision, Recall')
        parser.add_argument('--pre_gpu', type=int, default=0,
                            help='Whether put all batches to gpu before run batches. \
                            If 0, dynamically put gpu for each batch.')
        return parser

    def __init__(self, optimizer='GD', lr=0.01, epoch=100, batch_size=128, eval_batch_size=128 * 128,
                 dropout=0.2, l2=1e-5, l2s=1e-5, l2_bias=0,
                 grad_clip=10, metrics='RMSE', check_epoch=10, early_stop=1, pre_gpu=0):
        """
        初始化
        :param optimizer: 优化器名字
        :param lr: 学习率
        :param epoch: 总共跑几轮
        :param batch_size: 训练batch大小
        :param eval_batch_size: 测试batch大小
        :param dropout: dropout比例
        :param l2: l2权重
        :param metrics: 评价指标，逗号分隔
        :param check_epoch: 每几轮输出check一次模型中间的一些tensor
        :param early_stop: 是否自动提前终止训练
        
        Initialization
        :param optimizer: name of optimizer
        :param lr: learning rate
        :param epoch: how many epochs to run
        :param batch_size: training batch size
        :param eval_batch_size: testing batch size
        :param dropout: dropout ratio
        :param l2: wight of l2 regularizer
        :param metrics: evaluation metrics, seperated by comma
        :param check_epoch: every check_epoch rounds, output the intermediate result tensor of the model
        :param early_stop: if or not to do early stopping
        """
        self.optimizer_name = optimizer
        self.lr = lr
        self.epoch = epoch
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.dropout = dropout
        self.no_dropout = 0.0
        self.l2_weight = l2
        self.l2s_weight = l2s
        self.l2_bias = l2_bias
        self.grad_clip = grad_clip
        self.pre_gpu = pre_gpu

        # 把metrics转换为list of str
        # Convert metrics to list of str
        self.metrics = metrics.lower().split(',')
        self.check_epoch = check_epoch
        self.early_stop = early_stop
        self.time = None

        # 用来记录训练集、验证集、测试集每一轮的评价指标
        # Used to record the evaluation measures of training, validation and testing set in each round
        self.train_results, self.valid_results, self.test_results, self.loss, self.r_losses, self.r_p_n, \
        self.r_and_true, self.r_and_self, self.r_or_true, self.r_or_self = [], [], [], [], [], [], [], [], [], []

    def _build_optimizer(self, model):
        """
        创建优化器
        :param model: 模型
        :return: 优化器
        
        Create the optimizer
        :param model: model
        :return: optimizer
        """
        weight_p, bias_p = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if 'bias' in name:
                bias_p.append(p)
            else:
                weight_p.append(p)
        if self.l2_bias == 1:
            optimize_dict = [{'params': weight_p + bias_p, 'weight_decay': self.l2_weight}]
        else:
            optimize_dict = [{'params': weight_p, 'weight_decay': self.l2_weight},
                             {'params': bias_p, 'weight_decay': 0.0}]

        optimizer_name = self.optimizer_name.lower()
        if optimizer_name == 'gd':
            logging.info("Optimizer: GD")
            optimizer = torch.optim.SGD(optimize_dict, lr=self.lr)
        elif optimizer_name == 'adagrad':
            logging.info("Optimizer: Adagrad")
            optimizer = torch.optim.Adagrad(optimize_dict, lr=self.lr)
        elif optimizer_name == 'adam':
            logging.info("Optimizer: Adam")
            optimizer = torch.optim.Adam(optimize_dict, lr=self.lr)
        else:
            logging.error("Unknown Optimizer: " + self.optimizer_name)
            assert self.optimizer_name in ['GD', 'Adagrad', 'Adam']
            optimizer = torch.optim.SGD(optimize_dict, lr=self.lr)
        return optimizer

    def _check_time(self, start=False):
        """
        记录时间用，self.time保存了[起始时间，上一步时间]
        :param start: 是否开始计时
        :return: 上一步到当前位置的时间
        
        Record the time, self.time records [starting time, time of last step]
        :param start: if or not to start time counting
        :return: the time to reach current position in the previous step
        """
        if self.time is None or start:
            self.time = [t()] * 2
            return self.time[0]
        tmp_time = self.time[1]
        self.time[1] = t()
        return self.time[1] - tmp_time

    def batches_add_control(self, batches, train):
        """
        向所有batch添加一些控制信息比如DROPOUT
        :param batches: 所有batch的list，由DataProcessor产生
        :param train: 是否是训练阶段
        :return: 所有batch的list
        
        Add some control information into all batches, such as DROPOUT
        :param batches: list of all batches, produced by DataProcessor
        :param train: if or not this is training stage
        :return: list of all batches
        """
        for batch in batches:
            batch[TRAIN] = train
            batch[DROPOUT] = self.dropout if train else self.no_dropout
        return batches

    def predict(self, model, data, data_processor):
        """
        预测，不训练
        :param model: 模型
        :param data: 数据dict，由DataProcessor的self.get_*_data()和self.format_data_dict()系列函数产生
        :param data_processor: DataProcessor实例
        :return: prediction 拼接好的 np.array
        
        Predict, not training
        :param model: model
        :param data: data dict，produced by the self.get_*_data() and self.format_data_dict() function of DataProcessor
        :param data_processor: DataProcessor instance
        :return: prediction the concatenated np.array
        """
        gc.collect()

        batches = data_processor.prepare_batches(data, self.eval_batch_size, train=False, model=model)
        batches = self.batches_add_control(batches, train=False)
        if self.pre_gpu == 1:
            batches = [data_processor.batch_to_gpu(b) for b in batches]

        model.eval()
        predictions = []
        for batch in tqdm(batches, leave=False, ncols=100, mininterval=1, desc='Predict'):
            if self.pre_gpu == 0:
                batch = data_processor.batch_to_gpu(batch)
            prediction = model.predict(batch)[PREDICTION]
            predictions.append(prediction.detach().cpu().data.numpy())

        predictions = np.concatenate(predictions)
        sample_ids = np.concatenate([b[SAMPLE_ID] for b in batches])

        reorder_dict = dict(zip(sample_ids, predictions))
        predictions = np.array([reorder_dict[i] for i in data[SAMPLE_ID]])

        gc.collect()
        return predictions

    def fit(self, model, data, data_processor, epoch=-1):  # fit the results for an input set
        """
        训练
        :param model: 模型
        :param data: 数据dict，由DataProcessor的self.get_*_data()和self.format_data_dict()系列函数产生
        :param data_processor: DataProcessor实例
        :param epoch: 第几轮
        :return: 返回最后一轮的输出，可供self.check函数检查一些中间结果
        
        Training
        :param model: model
        :param data: data dict，produced by the self.get_*_data() and self.format_data_dict() function of DataProcessor
        :param data_processor: DataProcessor instance
        :param epoch: number of epoch
        :return: return the output of the last round，can be used by self.check function to check some intermediate results
        """
        gc.collect()
        if model.optimizer is None:
            model.optimizer = self._build_optimizer(model)
        batches = data_processor.prepare_batches(data, self.batch_size, train=True, model=model)
        batches = self.batches_add_control(batches, train=True)
        if self.pre_gpu == 1:
            batches = [data_processor.batch_to_gpu(b) for b in batches]

        batch_size = self.batch_size if data_processor.rank == 0 else self.batch_size * 2
        model.train()
        accumulate_size, prediction_list, output_dict = 0, [], None
        loss_list, loss_l2_list = [], []
        for i, batch in \
                tqdm(list(enumerate(batches)), leave=False, desc='Epoch %5d' % (epoch + 1), ncols=100, mininterval=1):
            if self.pre_gpu == 0:
                batch = data_processor.batch_to_gpu(batch)
            accumulate_size += len(batch[Y])
            model.optimizer.zero_grad()
            output_dict = model(batch)
            l2 = output_dict[LOSS_L2]
            loss = output_dict[LOSS] + l2 * self.l2s_weight
            loss.backward()
            loss_list.append(loss.detach().cpu().data.numpy())
            loss_l2_list.append(l2.detach().cpu().data.numpy())
            prediction_list.append(output_dict[PREDICTION].detach().cpu().data.numpy()[:batch[REAL_BATCH_SIZE]])
            if self.grad_clip > 0:
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 100)
                torch.nn.utils.clip_grad_value_(model.parameters(), self.grad_clip)
            if accumulate_size >= batch_size or i == len(batches) - 1:
                model.optimizer.step()
                accumulate_size = 0
            # model.optimizer.step()
        model.eval()
        gc.collect()

        predictions = np.concatenate(prediction_list)
        sample_ids = np.concatenate([b[SAMPLE_ID][:b[REAL_BATCH_SIZE]] for b in batches])
        reorder_dict = dict(zip(sample_ids, predictions))
        predictions = np.array([reorder_dict[i] for i in data[SAMPLE_ID]])
        return predictions, output_dict, np.mean(loss_list), np.mean(loss_l2_list)

    def eva_termination(self, model):
        """
        检查是否终止训练，基于验证集
        :param model: 模型
        :return: 是否终止训练
        
        Check if or not to stop training, based on validation set
        :param model: model
        :return: if or not to stop training
        """
        metric = self.metrics[0]
        valid = self.valid_results
        # 如果已经训练超过20轮，且评价指标越小越好，且评价已经连续五轮非减
        # If has been trained for over 20 rounds, and evaluation measure is the smaller the better, and the measure has been non-desceasing for five rounds
        if len(valid) > 20 and metric in utils.LOWER_METRIC_LIST and utils.strictly_increasing(valid[-5:]):
            return True
        # 如果已经训练超过20轮，且评价指标越大越好，且评价已经连续五轮非增
        # If has been trained for over 20 rounds, and evaluation measure is the larger the better, and the measure has been non-increasing for five rounds
        elif len(valid) > 20 and metric not in utils.LOWER_METRIC_LIST and utils.strictly_decreasing(valid[-5:]):
            return True
        # 训练好结果离当前已经20轮以上了
        # It has been more than 20 rounds from the best result
        elif len(valid) - valid.index(utils.best_result(metric, valid)) > 20:
            return True
        return False

    def train(self, model, data_processor):
        """
        训练模型
        :param model: 模型
        :param data_processor: DataProcessor实例
        :return:
        
        Model training
        :param model: model
        :param data_processor: DataProcessor instance
        :return:
        """

        # 获得训练、验证、测试数据，epoch=-1不shuffle
        # Obtain the training, validation and testing data, epoch=-1 no shuffling
        train_data = data_processor.get_train_data(epoch=-1, model=model)
        validation_data = data_processor.get_validation_data(model=model)
        test_data = data_processor.get_test_data(model=model) if data_processor.unlabel_test == 0 else None
        # 记录初始时间
        # Record start time
        self._check_time(start=True)  

        # 训练之前的模型效果
        # Model performance before training
        init_train = self.evaluate(model, train_data, data_processor) \
            if train_data is not None else [-1.0] * len(self.metrics)
        init_valid = self.evaluate(model, validation_data, data_processor) \
            if validation_data is not None else [-1.0] * len(self.metrics)
        init_test = self.evaluate(model, test_data, data_processor) \
            if test_data is not None and data_processor.unlabel_test == 0 else [-1.0] * len(self.metrics)
        logging.info("Init: \t train= %s validation= %s test= %s [%.1f s] " % (
            utils.format_metric(init_train), utils.format_metric(init_valid), utils.format_metric(init_test),
            self._check_time()) + ','.join(self.metrics))
        # model.save_model(
        #     model_path='../model/variable_tsne_logic_epoch/variable_tsne_logic_epoch_0.pt')
        try:
            for epoch in range(self.epoch):
                self._check_time()
                # 每一轮需要重新获得训练数据，因为涉及shuffle或者topn推荐时需要重新采样负例
                # Need to obtain training data again in each round, because it's related to shuffling or need to resample negative examples in topn recommendation
                epoch_train_data = data_processor.get_train_data(epoch=epoch, model=model)
                train_predictions, last_batch, mean_loss, mean_loss_l2 = \
                    self.fit(model, epoch_train_data, data_processor, epoch=epoch)

                # 检查模型中间结果
                # Check the intermediate results of the model
                if self.check_epoch > 0 and (epoch == 1 or epoch % self.check_epoch == 0):
                    last_batch['mean_loss'] = mean_loss
                    last_batch['mean_loss_l2'] = mean_loss_l2
                    self.check(model, last_batch)
                training_time = self._check_time()

                # # evaluate模型效果
                # # evaluate the model performance
                train_result = [mean_loss] + model.evaluate_method(train_predictions, train_data, metrics=['rmse'])
                valid_result = self.evaluate(model, validation_data, data_processor) \
                    if validation_data is not None else [-1.0] * len(self.metrics)
                test_result = self.evaluate(model, test_data, data_processor) \
                    if test_data is not None and data_processor.unlabel_test == 0 else [-1.0] * len(self.metrics)
                testing_time = self._check_time()

                self.train_results.append(train_result)
                self.valid_results.append(valid_result)
                self.test_results.append(test_result)
                self.loss.append(mean_loss)

                if 'r_loss' in last_batch:
                    self.r_losses.append(last_batch['r_loss'])
                if 'r_p_n' in last_batch:
                    self.r_p_n.append(last_batch['r_p_n'])
                if 'r_and_true' in last_batch:
                    self.r_and_true.append(last_batch['r_and_true'])
                if 'r_and_self' in last_batch:
                    self.r_and_self.append(last_batch['r_and_self'])
                if 'r_or_true' in last_batch:
                    self.r_or_true.append(last_batch['r_or_true'])
                if 'r_or_self' in last_batch:
                    self.r_or_self.append(last_batch['r_or_self'])

                # 输出当前模型效果
                # Output the current model performance
                logging.info("Epoch %5d [%.1f s]\t train= %s validation= %s test= %s [%.1f s] "
                             % (epoch + 1, training_time, utils.format_metric(train_result),
                                utils.format_metric(valid_result), utils.format_metric(test_result),
                                testing_time) + ','.join(self.metrics))

                # 如果当前效果是最优的，保存模型，基于验证集
                # If the current performance is the best, save the model, based on validate set
                if utils.best_result(self.metrics[0], self.valid_results) == self.valid_results[-1]:
                    model.save_model()
                # model.save_model(
                #     model_path='../model/variable_tsne_logic_epoch/variable_tsne_logic_epoch_%d.pt' % (epoch + 1))
                # 检查是否终止训练，基于验证集
                # Check if to stop training, based on validate set
                if self.eva_termination(model) and self.early_stop == 1:
                    logging.info("Early stop at %d based on validation result." % (epoch + 1))
                    break
        except KeyboardInterrupt:
            logging.info("Early stop manually")
            save_here = input("Save here? (1/0) (default 0):")
            if str(save_here).lower().startswith('1'):
                model.save_model()

        # #
        # xz = range(len(self.loss))
        # plt.rcParams['figure.figsize'] = (9.6, 5.8)
        # plt.rcParams['figure.dpi'] = 200
        # plt.rcParams['savefig.dpi'] = 200
        #
        # fig = plt.figure()
        # fig.subplots_adjust(hspace=0.3, wspace=0.5)
        # plt.subplot(2, 3, 1)
        # plt.plot(xz, self.loss, color='k')
        # plt.title('loss')
        # plt.grid(linestyle="--")
        # ax = plt.gca()
        # ax.spines['top'].set_visible(False)  # 去掉上边框
        # ax.spines['right'].set_visible(False)
        # plt.subplot(2, 3, 2)
        # plt.plot(xz, self.test_results, color='k')
        # plt.title('result')
        # plt.grid(linestyle="--")
        # ax = plt.gca()
        # ax.spines['top'].set_visible(False)  # 去掉上边框
        # ax.spines['right'].set_visible(False)
        # if 'r_loss' in last_batch:
        #     plt.subplot(2, 3, 3)
        #     plt.plot(xz, self.r_losses, color='k')
        #     plt.title('r_loss')
        #     plt.grid(linestyle="--")
        #     ax = plt.gca()
        #     ax.spines['top'].set_visible(False)  # 去掉上边框
        #     ax.spines['right'].set_visible(False)
        # if 'r_p_n' in last_batch:
        #     plt.subplot(2, 3, 4)
        #     plt.plot(xz, self.r_p_n, color='k')
        #     plt.title('r_p_n')
        #     plt.grid(linestyle="--")
        #     ax = plt.gca()
        #     ax.spines['top'].set_visible(False)  # 去掉上边框
        #     ax.spines['right'].set_visible(False)
        # if 'r_and_true' in last_batch:
        #     plt.subplot(2, 4, 5)
        #     plt.plot(xz, self.r_and_true, color='k')
        #     plt.title('r_and_true')
        #     plt.grid(linestyle="--")
        #     ax = plt.gca()
        #     ax.spines['top'].set_visible(False)  # 去掉上边框
        #     ax.spines['right'].set_visible(False)
        # if 'r_and_self' in last_batch:
        #     plt.subplot(2, 4, 6)
        #     plt.plot(xz, self.r_and_self, color='k')
        #     plt.title('r_and_self')
        #     plt.grid(linestyle="--")
        #     ax = plt.gca()
        #     ax.spines['top'].set_visible(False)  # 去掉上边框
        #     ax.spines['right'].set_visible(False)
        # if 'r_or_true' in last_batch:
        #     plt.subplot(2, 3, 6)
        #     plt.plot(xz, self.r_or_true, color='k')
        #     plt.title('r_or_true')
        #     plt.grid(linestyle="--")
        #     ax = plt.gca()
        #     ax.spines['top'].set_visible(False)  # 去掉上边框
        #     ax.spines['right'].set_visible(False)
        # if 'r_or_self' in last_batch:
        #     plt.subplot(2, 3, 5)
        #     plt.plot(xz, self.r_or_self, color='k')
        #     plt.title('r_or_self')
        #     plt.grid(linestyle="--")
        #     ax = plt.gca()
        #     ax.spines['top'].set_visible(False)  # 去掉上边框
        #     ax.spines['right'].set_visible(False)
        # date_str = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
        # plt.savefig('./' + date_str + '.png')
        # #

        # Find the best validation result across iterations
        best_valid_score = utils.best_result(self.metrics[0], self.valid_results)
        best_epoch = self.valid_results.index(best_valid_score)
        logging.info("Best Iter(validation)= %5d\t train= %s valid= %s test= %s [%.1f s] "
                     % (best_epoch + 1,
                        utils.format_metric(self.train_results[best_epoch]),
                        utils.format_metric(self.valid_results[best_epoch]),
                        utils.format_metric(self.test_results[best_epoch]),
                        self.time[1] - self.time[0]) + ','.join(self.metrics))
        best_test_score = utils.best_result(self.metrics[0], self.test_results)
        best_epoch = self.test_results.index(best_test_score)
        logging.info("Best Iter(test)= %5d\t train= %s valid= %s test= %s [%.1f s] "
                     % (best_epoch + 1,
                        utils.format_metric(self.train_results[best_epoch]),
                        utils.format_metric(self.valid_results[best_epoch]),
                        utils.format_metric(self.test_results[best_epoch]),
                        self.time[1] - self.time[0]) + ','.join(self.metrics))
        model.load_model()

    def evaluate(self, model, data, data_processor, metrics=None):  # evaluate the results for an input set
        """
        evaluate模型效果
        :param model: 模型
        :param data: 数据dict，由DataProcessor的self.get_*_data()和self.format_data_dict()系列函数产生
        :param data_processor: DataProcessor
        :param metrics: list of str
        :return: list of float 每个对应一个 metric
        
        evaluate the model performance
        :param model: model
        :param data: data dict，produced by the self.get_*_data() and self.format_data_dict() function of DataProcessor
        :param data_processor: DataProcessor
        :param metrics: list of str
        :return: list of float, each corresponding to a metric
        """
        if metrics is None:
            metrics = self.metrics
        predictions = self.predict(model, data, data_processor)
        return model.evaluate_method(predictions, data, metrics=metrics)

    def check(self, model, out_dict):
        """
        检查模型中间结果
        :param model: 模型
        :param out_dict: 某一个batch的模型输出结果
        :return:
        
        Check the intermediate result of the model
        :param model: model
        :param out_dict: model output of a certain batch
        :return:
        """
        # batch = data_processor.get_feed_dict(data, 0, self.batch_size, True)
        # self.batches_add_control([batch], train=False)
        # model.eval()
        # check = model(batch)
        check = out_dict
        logging.info(os.linesep)
        for i, t in enumerate(check[CHECK]):
            d = np.array(t[1].detach().cpu())
            logging.info(os.linesep.join([t[0] + '\t' + str(d.shape), np.array2string(d, threshold=20)]) + os.linesep)

        loss, l2 = check['mean_loss'], check['mean_loss_l2']
        logging.info('mean loss = %.4f, l2 = %.4f, %.4f' % (loss, l2 * self.l2_weight, l2 * self.l2s_weight))
        # if not (loss * 0.005 < l2 < loss * 0.1):
        #     logging.warning('l2 inappropriate: loss = %.4f, l2 = %.4f' % (loss, l2))

    def run_some_tensors(self, model, data, data_processor, dict_keys):
        """
        预测，不训练
        :param model: 模型
        :param data: 数据dict，由DataProcessor的self.get_*_data()和self.format_data_dict()系列函数产生
        :param data_processor: DataProcessor实例
        :return: prediction 拼接好的 np.array
        
        Predict, no training
        :param model: model
        :param data: data dict，produced by the self.get_*_data() and self.format_data_dict() functions of DataProcessor
        :param data_processor: DataProcessor instance
        :return: prediction, concatenated np.array
        """
        gc.collect()

        if type(dict_keys) == str:
            dict_keys = [dict_keys]

        batches = data_processor.prepare_batches(data, self.eval_batch_size, train=False, model=model)
        batches = self.batches_add_control(batches, train=False)
        if self.pre_gpu == 1:
            batches = [data_processor.batch_to_gpu(b) for b in batches]

        result_dict = {}
        for key in dict_keys:
            result_dict[key] = []
        model.eval()
        for batch in tqdm(batches, leave=False, ncols=100, mininterval=1, desc='Predict'):
            if self.pre_gpu == 0:
                batch = data_processor.batch_to_gpu(batch)
            out_dict = model.predict(batch)
            for key in dict_keys:
                if key in out_dict:
                    result_dict[key].append(out_dict[key].detach().cpu().data.numpy())

        sample_ids = np.concatenate([b[SAMPLE_ID] for b in batches])
        for key in dict_keys:
            try:
                result_array = np.concatenate(result_dict[key])
            except ValueError as e:
                logging.warning("run_some_tensors: %s %s" % (key, str(e)))
                result_array = np.array([d for b in result_dict[key] for d in b])
            if len(sample_ids) == len(result_array):
                reorder_dict = dict(zip(sample_ids, result_array))
                result_dict[key] = np.array([reorder_dict[i] for i in data[SAMPLE_ID]])
        gc.collect()
        return result_dict
